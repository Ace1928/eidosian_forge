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
def get_log_likelihood(idata, var_name=None, single_var=True):
    """Retrieve the log likelihood dataarray of a given variable."""
    if not hasattr(idata, 'log_likelihood') and hasattr(idata, 'sample_stats') and hasattr(idata.sample_stats, 'log_likelihood'):
        warnings.warn('Storing the log_likelihood in sample_stats groups has been deprecated', DeprecationWarning)
        return idata.sample_stats.log_likelihood
    if not hasattr(idata, 'log_likelihood'):
        raise TypeError('log likelihood not found in inference data object')
    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            if single_var:
                raise TypeError(f'Found several log likelihood arrays {var_names}, var_name cannot be None')
            return idata.log_likelihood[var_names]
        return idata.log_likelihood[var_names[0]]
    else:
        try:
            log_likelihood = idata.log_likelihood[var_name]
        except KeyError as err:
            raise TypeError(f'No log likelihood data named {var_name} found') from err
        return log_likelihood