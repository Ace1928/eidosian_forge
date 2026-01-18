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
def is_valid_quantile(value):
    """Check if value is a number between 0 and 1."""
    try:
        value = float(value)
        return 0 < value < 1
    except ValueError:
        return False