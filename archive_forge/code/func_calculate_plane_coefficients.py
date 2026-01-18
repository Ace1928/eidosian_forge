import sys
import numpy as np
from matplotlib import _api
def calculate_plane_coefficients(self, z):
    """
        Calculate plane equation coefficients for all unmasked triangles from
        the point (x, y) coordinates and specified z-array of shape (npoints).
        The returned array has shape (npoints, 3) and allows z-value at (x, y)
        position in triangle tri to be calculated using
        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
        """
    return self.get_cpp_triangulation().calculate_plane_coefficients(z)