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
def compute_ranks(ary):
    """Compute ranks for continuous and discrete variables."""
    if ary.dtype.kind == 'i':
        ary_shape = ary.shape
        ary = ary.flatten()
        min_ary, max_ary = (min(ary), max(ary))
        x = np.linspace(min_ary, max_ary, len(ary))
        csi = CubicSpline(x, ary)
        ary = csi(np.linspace(min_ary + 0.001, max_ary - 0.001, len(ary))).reshape(ary_shape)
    ranks = rankdata(ary, method='average').reshape(ary.shape)
    return ranks