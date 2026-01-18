import math
import warnings
from typing import Any, Optional, Union
from pyproj._geod import Geod as _Geod
from pyproj._geod import GeodIntermediateReturn, geodesic_version_str
from pyproj._geod import reverse_azimuth as _reverse_azimuth
from pyproj.enums import GeodIntermediateFlag
from pyproj.exceptions import GeodError
from pyproj.list import get_ellps_map
from pyproj.utils import DataType, _convertback, _copytobuffer
def geometry_area_perimeter(self, geometry, radians: bool=False) -> tuple[float, float]:
    """
        .. versionadded:: 2.3.0

        A simple interface for computing the area (meters^2) and perimeter (meters)
        of a geodesic polygon as a shapely geometry.

        Arbitrarily complex polygons are allowed.  In the case self-intersecting
        of polygons the area is accumulated "algebraically", e.g., the areas of
        the 2 loops in a figure-8 polygon will partially cancel.  There's no need
        to "close" the polygon by repeating the first vertex.

        .. note:: lats should be in the range [-90 deg, 90 deg].

        .. warning:: The area returned is signed with counter-clockwise (CCW) traversal
                     being treated as positive. For polygons, holes should use the
                     opposite traversal to the exterior (if the exterior is CCW, the
                     holes/interiors should be CW). You can use `shapely.ops.orient` to
                     modify the orientation.

        If it is a Polygon, it will return the area and exterior perimeter.
        It will subtract the area of the interior holes.
        If it is a MultiPolygon or MultiLine, it will return
        the sum of the areas and perimeters of all geometries.


        Example usage:

        >>> from pyproj import Geod
        >>> from shapely.geometry import LineString, Point, Polygon
        >>> geod = Geod(ellps="WGS84")
        >>> poly_area, poly_perimeter = geod.geometry_area_perimeter(
        ...     Polygon(
        ...         LineString([
        ...             Point(1, 1), Point(10, 1), Point(10, 10), Point(1, 10)
        ...         ]),
        ...         holes=[LineString([Point(1, 2), Point(3, 4), Point(5, 2)])],
        ...     )
        ... )
        >>> f"{poly_area:.0f} {poly_perimeter:.0f}"
        '944373881400 3979008'


        Parameters
        ----------
        geometry: :class:`shapely.geometry.BaseGeometry`
            The geometry to calculate the area and perimeter from.
        radians: bool, default=False
            If True, the input data is assumed to be in radians.
            Otherwise, the data is assumed to be in degrees.

        Returns
        -------
        (float, float):
            The geodesic area (meters^2) and perimeter (meters) of the polygon.
        """
    try:
        return self.polygon_area_perimeter(*geometry.xy, radians=radians)
    except (AttributeError, NotImplementedError):
        pass
    if hasattr(geometry, 'exterior'):
        total_area, total_perimeter = self.geometry_area_perimeter(geometry.exterior, radians=radians)
        for hole in geometry.interiors:
            area, _ = self.geometry_area_perimeter(hole, radians=radians)
            total_area += area
        return (total_area, total_perimeter)
    if hasattr(geometry, 'geoms'):
        total_area = 0.0
        total_perimeter = 0.0
        for geom in geometry.geoms:
            area, perimeter = self.geometry_area_perimeter(geom, radians=radians)
            total_area += area
            total_perimeter += perimeter
        return (total_area, total_perimeter)
    raise GeodError('Invalid geometry provided.')