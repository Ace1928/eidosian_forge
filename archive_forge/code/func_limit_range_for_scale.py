import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
def limit_range_for_scale(self, vmin, vmax, minpos):
    """
        Limit the domain to values between 0 and 1 (excluded).
        """
    if not np.isfinite(minpos):
        minpos = 1e-07
    return (minpos if vmin <= 0 else vmin, 1 - minpos if vmax >= 1 else vmax)