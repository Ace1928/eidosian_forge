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
def set_theta_offset(self, offset):
    """
        Set the offset for the location of 0 in radians.
        """
    mtx = self._theta_offset.get_matrix()
    mtx[0, 2] = offset
    self._theta_offset.invalidate()