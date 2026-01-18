import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
@array_function_dispatch(_structured_to_unstructured_dispatcher)
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    """
    Converts an n-D structured array into an (n+1)-D unstructured array.

    The new array will have a new last dimension equal in size to the
    number of field-elements of the input array. If not supplied, the output
    datatype is determined from the numpy type promotion rules applied to all
    the field datatypes.

    Nested fields, as well as each element of any subarray fields, all count
    as a single field-elements.

    Parameters
    ----------
    arr : ndarray
       Structured array or dtype to convert. Cannot contain object datatype.
    dtype : dtype, optional
       The dtype of the output unstructured array.
    copy : bool, optional
        If true, always return a copy. If false, a view is returned if
        possible, such as when the `dtype` and strides of the fields are
        suitable and the array subtype is one of `np.ndarray`, `np.recarray`
        or `np.memmap`.

        .. versionchanged:: 1.25.0
            A view can now be returned if the fields are separated by a
            uniform stride.

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `numpy.ndarray.astype`. Controls what kind of
        data casting may occur.

    Returns
    -------
    unstructured : ndarray
       Unstructured array with one more dimension.

    Examples
    --------

    >>> from numpy.lib import recfunctions as rfn
    >>> a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
    >>> a
    array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),
           (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],
          dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])
    >>> rfn.structured_to_unstructured(a)
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
    ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
    >>> np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)
    array([ 3. ,  5.5,  9. , 11. ])

    """
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')
    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    if n_fields == 0 and dtype is None:
        raise ValueError('arr has no fields. Unable to guess dtype')
    elif n_fields == 0:
        raise NotImplementedError('arr with no fields is not supported')
    dts, counts, offsets = zip(*fields)
    names = ['f{}'.format(n) for n in range(n_fields)]
    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = np.dtype(dtype)
    flattened_fields = np.dtype({'names': names, 'formats': dts, 'offsets': offsets, 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)
    can_view = type(arr) in (np.ndarray, np.recarray, np.memmap)
    if not copy and can_view and all((dt.base == out_dtype for dt in dts)):
        common_stride = _common_stride(offsets, counts, out_dtype.itemsize)
        if common_stride is not None:
            wrap = arr.__array_wrap__
            new_shape = arr.shape + (sum(counts), out_dtype.itemsize)
            new_strides = arr.strides + (abs(common_stride), 1)
            arr = arr[..., np.newaxis].view(np.uint8)
            arr = arr[..., min(offsets):]
            arr = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides, subok=True)
            arr = arr.view(out_dtype)[..., 0]
            if common_stride < 0:
                arr = arr[..., ::-1]
            if type(arr) is not type(wrap.__self__):
                arr = wrap(arr)
            return arr
    packed_fields = np.dtype({'names': names, 'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)
    return arr.view((out_dtype, (sum(counts),)))