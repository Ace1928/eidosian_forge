from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
def shapeup(self, ob):
    if isinstance(ob, BaseGeometry):
        return ob
    else:
        try:
            return shape(ob)
        except (ValueError, AttributeError):
            return LineString(ob)