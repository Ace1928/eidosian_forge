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
def set_longitude_grid(self, degrees):
    """
        Set the number of degrees between each longitude grid.
        """
    grid = np.arange(-180 + degrees, 180, degrees)
    self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
    self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))