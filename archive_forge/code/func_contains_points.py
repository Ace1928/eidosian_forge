import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def contains_points(self, points, transform=None, radius=0.0):
    """
        Return whether the area enclosed by the path contains the given points.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
    if transform is not None:
        transform = transform.frozen()
    result = _path.points_in_path(points, radius, self, transform)
    return result.astype('bool')