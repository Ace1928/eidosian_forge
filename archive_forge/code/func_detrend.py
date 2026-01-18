import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> (signal.detrend(x) - noise).max()
    0.06  # random

    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, N))))
        if np.any(bp > N):
            raise ValueError('Breakpoints must be less than length of data along given axis.')
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = np.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)
        if not overwrite_data:
            newdata = newdata.copy()
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - A @ coef
        newdata = newdata.reshape(newdata_shape)
        ret = np.moveaxis(newdata, 0, axis)
        return ret