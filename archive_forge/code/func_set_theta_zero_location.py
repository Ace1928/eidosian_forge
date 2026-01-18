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
def set_theta_zero_location(self, loc, offset=0.0):
    """
        Set the location of theta's zero.

        This simply calls `set_theta_offset` with the correct value in radians.

        Parameters
        ----------
        loc : str
            May be one of "N", "NW", "W", "SW", "S", "SE", "E", or "NE".
        offset : float, default: 0
            An offset in degrees to apply from the specified *loc*. **Note:**
            this offset is *always* applied counter-clockwise regardless of
            the direction setting.
        """
    mapping = {'N': np.pi * 0.5, 'NW': np.pi * 0.75, 'W': np.pi, 'SW': np.pi * 1.25, 'S': np.pi * 1.5, 'SE': np.pi * 1.75, 'E': 0, 'NE': np.pi * 0.25}
    return self.set_theta_offset(mapping[loc] + np.deg2rad(offset))