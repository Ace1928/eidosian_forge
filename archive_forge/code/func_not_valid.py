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
def not_valid(ary, check_nan=True, check_shape=True, nan_kwargs=None, shape_kwargs=None):
    """Validate ndarray.

    Parameters
    ----------
    ary : numpy.ndarray
    check_nan : bool
        Check if any value contains NaN.
    check_shape : bool
        Check if array has correct shape. Assumes dimensions in order (chain, draw, *shape).
        For 1D arrays (shape = (n,)) assumes chain equals 1.
    nan_kwargs : dict
        Valid kwargs are:
            axis : int,
                Defaults to None.
            how : str, {"all", "any"}
                Default to "any".
    shape_kwargs : dict
        Valid kwargs are:
            min_chains : int
                Defaults to 1.
            min_draws : int
                Defaults to 4.

    Returns
    -------
    bool
    """
    ary = np.asarray(ary)
    nan_error = False
    draw_error = False
    chain_error = False
    if check_nan:
        if nan_kwargs is None:
            nan_kwargs = {}
        isnan = np.isnan(ary)
        axis = nan_kwargs.get('axis', None)
        if nan_kwargs.get('how', 'any').lower() == 'all':
            nan_error = isnan.all(axis)
        else:
            nan_error = isnan.any(axis)
        if isinstance(nan_error, bool) and nan_error or nan_error.any():
            _log.warning('Array contains NaN-value.')
    if check_shape:
        shape = ary.shape
        if shape_kwargs is None:
            shape_kwargs = {}
        min_chains = shape_kwargs.get('min_chains', 2)
        min_draws = shape_kwargs.get('min_draws', 4)
        error_msg = f'Shape validation failed: input_shape: {shape}, '
        error_msg += f'minimum_shape: (chains={min_chains}, draws={min_draws})'
        chain_error = min_chains > 1 and len(shape) < 2 or shape[0] < min_chains
        draw_error = len(shape) < 2 and shape[0] < min_draws or (len(shape) > 1 and shape[1] < min_draws)
        if chain_error or draw_error:
            _log.warning(error_msg)
    return nan_error | chain_error | draw_error