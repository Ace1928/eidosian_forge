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
def smooth_data(obs_vals, pp_vals):
    """Smooth data using a cubic spline.

    Helper function for discrete data in plot_pbv, loo_pit and plot_loo_pit.

    Parameters
    ----------
    obs_vals : (N) array-like
        Observed data
    pp_vals : (S, N) array-like
        Posterior predictive samples. ``N`` is the number of observations,
        and ``S`` is the number of samples (generally n_chains*n_draws).

    Returns
    -------
    obs_vals : (N) ndarray
        Smoothed observed data
    pp_vals : (S, N) ndarray
        Smoothed posterior predictive samples
    """
    x = np.linspace(0, 1, len(obs_vals))
    csi = CubicSpline(x, obs_vals)
    obs_vals = csi(np.linspace(0.01, 0.99, len(obs_vals)))
    x = np.linspace(0, 1, pp_vals.shape[1])
    csi = CubicSpline(x, pp_vals, axis=1)
    pp_vals = csi(np.linspace(0.01, 0.99, pp_vals.shape[1]))
    return (obs_vals, pp_vals)