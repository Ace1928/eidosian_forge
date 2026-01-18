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
def set_rorigin(self, rorigin):
    """
        Update the radial origin.

        Parameters
        ----------
        rorigin : float
        """
    self._originViewLim.locked_y0 = rorigin