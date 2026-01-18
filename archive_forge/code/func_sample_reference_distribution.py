import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def sample_reference_distribution(dist, shape):
    """Generate samples from a scipy distribution with a given shape."""
    x_ss = []
    densities = []
    dist_rvs = dist.rvs(size=shape)
    for idx in range(shape[1]):
        x_s, density = kde(dist_rvs[:, idx])
        x_ss.append(x_s)
        densities.append(density)
    return (np.array(x_ss).T, np.array(densities).T)