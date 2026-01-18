import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def autocov(ary, axis=-1):
    """Compute autocovariance estimates for every lag for the input array.

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    axis = axis if axis > 0 else len(ary.shape) + axis
    n = ary.shape[axis]
    m = next_fast_len(2 * n)
    ary = ary - ary.mean(axis, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ifft_ary = np.fft.rfft(ary, n=m, axis=axis)
        ifft_ary *= np.conjugate(ifft_ary)
        shape = tuple((slice(None) if dim_len != axis else slice(0, n) for dim_len, _ in enumerate(ary.shape)))
        cov = np.fft.irfft(ifft_ary, n=m, axis=axis)[shape]
        cov /= n
    return cov