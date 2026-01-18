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
def get_theta_direction(self):
    """
        Get the direction in which theta increases.

        -1:
           Theta increases in the clockwise direction

        1:
           Theta increases in the counterclockwise direction
        """
    return self._direction.get_matrix()[0, 0]