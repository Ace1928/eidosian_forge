from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
def cascaded_union(self, geoms):
    """Returns the union of a sequence of geometries

        .. deprecated:: 1.8
            This function was superseded by :meth:`unary_union`.
        """
    warn("The 'cascaded_union()' function is deprecated. Use 'unary_union()' instead.", ShapelyDeprecationWarning, stacklevel=2)
    return shapely.union_all(geoms, axis=None)