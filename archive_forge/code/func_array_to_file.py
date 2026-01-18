from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def array_to_file(data: npt.ArrayLike, fileobj: io.IOBase, out_dtype: np.dtype | None=None, offset: int=0, intercept: Scalar=0.0, divslope: Scalar | None=1.0, mn: Scalar | None=None, mx: Scalar | None=None, order: ty.Literal['C', 'F']='F', nan2zero: bool=True) -> None:
    """Helper function for writing arrays to file objects

    Writes arrays as scaled by `intercept` and `divslope`, and clipped
    at (prescaling) `mn` minimum, and `mx` maximum.

    * Clip `data` array at min `mn`, max `max` where there are not None ->
      ``clipped`` (this is *pre scale clipping*)
    * Scale ``clipped`` with ``clipped_scaled = (clipped - intercept) /
      divslope``
    * Clip ``clipped_scaled`` to fit into range of `out_dtype` (*post scale
      clipping*) -> ``clipped_scaled_clipped``
    * If converting to integer `out_dtype` and `nan2zero` is True, set NaN
      values in ``clipped_scaled_clipped`` to 0
    * Write ``clipped_scaled_clipped_n2z`` to fileobj `fileobj` starting at
      offset `offset` in memory layout `order`

    Parameters
    ----------
    data : array-like
        array or array-like to write.
    fileobj : file-like
        file-like object implementing ``write`` method.
    out_dtype : None or dtype, optional
        dtype to write array as.  Data array will be coerced to this dtype
        before writing. If None (default) then use input data type.
    offset : None or int, optional
        offset into fileobj at which to start writing data. Default is 0. None
        means start at current file position
    intercept : scalar, optional
        scalar to subtract from data, before dividing by ``divslope``.  Default
        is 0.0
    divslope : None or scalar, optional
        scalefactor to *divide* data by before writing.  Default is 1.0. If
        None, there is no valid data, we write zeros.
    mn : scalar, optional
        minimum threshold in (unscaled) data, such that all data below this
        value are set to this value. Default is None (no threshold). The
        typical use is to set -np.inf in the data to have this value (which
        might be the minimum non-finite value in the data).
    mx : scalar, optional
        maximum threshold in (unscaled) data, such that all data above this
        value are set to this value. Default is None (no threshold). The
        typical use is to set np.inf in the data to have this value (which
        might be the maximum non-finite value in the data).
    order : {'F', 'C'}, optional
        memory order to write array.  Default is 'F'
    nan2zero : {True, False}, optional
        Whether to set NaN values to 0 when writing integer output.  Defaults
        to True.  If False, NaNs will be represented as numpy does when
        casting; this depends on the underlying C library and is undefined. In
        practice `nan2zero` == False might be a good choice when you completely
        sure there will be no NaNs in the data. This value ignored for float
        output types.  NaNs are treated as zero *before* applying `intercept`
        and `divslope` - so an array ``[np.nan]`` with an `intercept` of 10
        becomes ``[-10]`` after conversion to integer `out_dtype` with
        `nan2zero` set.  That is because you will likely apply `divslope` and
        `intercept` in reverse order when reading the data back, returning the
        zero you probably expected from the input NaN.

    Examples
    --------
    >>> from io import BytesIO
    >>> sio = BytesIO()
    >>> data = np.arange(10, dtype=np.float64)
    >>> array_to_file(data, sio, np.float64)
    >>> sio.getvalue() == data.tobytes('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)  # outputs 0
    >>> array_to_file(data, sio, np.int16)
    >>> sio.getvalue() == data.astype(np.int16).tobytes()
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data.byteswap(), sio, np.float64)
    >>> sio.getvalue() == data.byteswap().tobytes('F')
    True
    >>> _ = sio.truncate(0); _ = sio.seek(0)
    >>> array_to_file(data, sio, np.float64, order='C')
    >>> sio.getvalue() == data.tobytes('C')
    True
    """
    if not np.isfinite(np.array((intercept, 1.0 if divslope is None else divslope))).all():
        raise ValueError('divslope and intercept must be finite')
    if divslope == 0:
        raise ValueError('divslope cannot be zero')
    data = np.asanyarray(data)
    in_dtype = data.dtype
    if out_dtype is None:
        out_dtype = in_dtype
    else:
        out_dtype = np.dtype(out_dtype)
    if offset is not None:
        seek_tell(fileobj, offset)
    if divslope is None or (mn, mx) == (0, 0) or ((mn is not None and mx is not None) and mx < mn):
        write_zeros(fileobj, data.size * out_dtype.itemsize)
        return
    if order not in 'FC':
        raise ValueError('Order should be one of F or C')
    pre_clips = None if mn is None and mx is None else (mn, mx)
    null_scaling = intercept == 0 and divslope == 1
    if in_dtype.type == np.void:
        if not null_scaling:
            raise ValueError('Cannot scale non-numeric types')
        if pre_clips is not None:
            raise ValueError('Cannot clip non-numeric types')
        return _write_data(data, fileobj, out_dtype, order)
    if pre_clips is not None:
        pre_clips = _dt_min_max(in_dtype, *pre_clips)
    if null_scaling and np.can_cast(in_dtype, out_dtype):
        return _write_data(data, fileobj, out_dtype, order, pre_clips=pre_clips)
    slope, inter = (np.atleast_1d(v) for v in (divslope, intercept))
    if slope.dtype.kind in 'iu':
        slope = slope.astype(float)
    if inter.dtype.kind in 'iu':
        inter = inter.astype(float)
    in_kind = in_dtype.kind
    out_kind = out_dtype.kind
    if out_kind in 'fc':
        return _write_data(data, fileobj, out_dtype, order, slope=slope, inter=inter, pre_clips=pre_clips)
    assert out_kind in 'iu'
    if in_kind in 'iu':
        if null_scaling:
            mn, mx = _dt_min_max(in_dtype, mn, mx)
            mn_out, mx_out = _dt_min_max(out_dtype)
            pre_clips = (max(mn, mn_out), min(mx, mx_out))
            return _write_data(data, fileobj, out_dtype, order, pre_clips=pre_clips)
        nan2zero = False
    slope, inter = (v.astype(_matching_float(v.dtype)) for v in (slope, inter))
    pre_clips = None
    cast_in_dtype = in_dtype
    if in_kind == 'c':
        cast_in_dtype = np.dtype(_matching_float(in_dtype))
    elif in_kind == 'f' and in_dtype.itemsize == 2:
        cast_in_dtype = np.dtype(np.float32)
    w_type = working_type(cast_in_dtype, slope, inter)
    dt_mnmx = _dt_min_max(cast_in_dtype, mn, mx)
    extremes = np.array(dt_mnmx, dtype=cast_in_dtype)
    w_type = best_write_scale_ftype(extremes, slope, inter, w_type)
    slope, inter = (v.astype(w_type) for v in (slope, inter))
    specials = np.array(dt_mnmx + (0,), dtype=w_type)
    if inter != 0.0:
        specials = specials - inter
    if slope != 1.0:
        specials = specials / slope
    assert specials.dtype.type == w_type
    post_mn, post_mx, nan_fill = np.rint(specials)
    if post_mn > post_mx:
        post_mn, post_mx = (post_mx, post_mn)
    both_mn, both_mx = shared_range(w_type, out_dtype)
    if nan2zero and (not both_mn <= nan_fill <= both_mx):
        est_err = np.round(2 * np.finfo(w_type).eps * abs(inter / slope))
        if nan_fill < both_mn and abs(nan_fill - both_mn) < est_err or (nan_fill > both_mx and abs(nan_fill - both_mx) < est_err):
            nan_fill = np.clip(nan_fill, both_mn, both_mx)
        else:
            raise ValueError(f'nan_fill == {nan_fill}, outside safe int range ({int(both_mn)}-{int(both_mx)}); change scaling or set nan2zero=False?')
    post_mn = np.max([post_mn, both_mn])
    post_mx = np.min([post_mx, both_mx])
    in_cast = None if cast_in_dtype == in_dtype else cast_in_dtype
    return _write_data(data, fileobj, out_dtype, order, in_cast=in_cast, pre_clips=pre_clips, inter=inter, slope=slope, post_clips=(post_mn, post_mx), nan_fill=nan_fill if nan2zero else None)