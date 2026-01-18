import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
def nonsingular(self, vmin, vmax):
    if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
        return (0, 1)
    else:
        return self.base.nonsingular(vmin, vmax)