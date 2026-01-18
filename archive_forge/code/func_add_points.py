import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def add_points(self, points, meth='s'):
    """Add a list of data points to Bloch sphere.

        Args:
            points (array_like):
                Collection of data points.
            meth (str):
                Type of points to plot, use 'm' for multicolored, 'l' for points
                connected with a line.
        """
    if not isinstance(points[0], (list, np.ndarray)):
        points = [[points[0]], [points[1]], [points[2]]]
    points = np.array(points)
    if meth == 's':
        if len(points[0]) == 1:
            pnts = np.array([[points[0][0]], [points[1][0]], [points[2][0]]])
            pnts = np.append(pnts, points, axis=1)
        else:
            pnts = points
        self.points.append(pnts)
        self.point_style.append('s')
    elif meth == 'l':
        self.points.append(points)
        self.point_style.append('l')
    else:
        self.points.append(points)
        self.point_style.append('m')