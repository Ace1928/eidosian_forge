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
def get_rlabel_position(self):
    """
        Returns
        -------
        float
            The theta position of the radius labels in degrees.
        """
    return np.rad2deg(self._r_label_position.get_matrix()[0, 2])