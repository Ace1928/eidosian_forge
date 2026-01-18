import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree
def calculate_areas(self):
    """Calculates the areas of the Voronoi regions.

        For 2D point sets, the regions are circular arcs. The sum of the areas
        is `2 * pi * radius`.

        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is `4 * pi * radius**2`.

        .. versionadded:: 1.5.0

        Returns
        -------
        areas : double array of shape (npoints,)
            The areas of the Voronoi regions.
        """
    if self._dim == 2:
        return self._calculate_areas_2d()
    elif self._dim == 3:
        return self._calculate_areas_3d()
    else:
        raise TypeError('Only supported for 2D and 3D point sets')