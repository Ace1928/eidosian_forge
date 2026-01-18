import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
def set_latitude_grid(self, degrees):
    """
        Set the number of degrees between each latitude grid.
        """
    grid = np.arange(-90 + degrees, 90, degrees)
    self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
    self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))