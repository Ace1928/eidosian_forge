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
def polygon_area_perimeter(self, lons: Any, lats: Any, radians: bool=False) -> tuple[float, float]:
    """
        .. versionadded:: 2.3.0

        A simple interface for computing the area (meters^2) and perimeter (meters)
        of a geodesic polygon.

        Arbitrarily complex polygons are allowed. In the case self-intersecting
        of polygons the area is accumulated "algebraically", e.g., the areas of
        the 2 loops in a figure-8 polygon will partially cancel. There's no need
        to "close" the polygon by repeating the first vertex. The area returned
        is signed with counter-clockwise traversal being treated as positive.

        .. note:: lats should be in the range [-90 deg, 90 deg].


        Example usage:

        >>> from pyproj import Geod
        >>> geod = Geod('+a=6378137 +f=0.0033528106647475126')
        >>> lats = [-72.9, -71.9, -74.9, -74.3, -77.5, -77.4, -71.7, -65.9, -65.7,
        ...         -66.6, -66.9, -69.8, -70.0, -71.0, -77.3, -77.9, -74.7]
        >>> lons = [-74, -102, -102, -131, -163, 163, 172, 140, 113,
        ...         88, 59, 25, -4, -14, -33, -46, -61]
        >>> poly_area, poly_perimeter = geod.polygon_area_perimeter(lons, lats)
        >>> f"{poly_area:.1f} {poly_perimeter:.1f}"
        '13376856682207.4 14710425.4'


        Parameters
        ----------
        lons: array, :class:`numpy.ndarray`, list, tuple, or scalar
            An array of longitude values.
        lats: array, :class:`numpy.ndarray`, list, tuple, or scalar
            An array of latitude values.
        radians: bool, default=False
            If True, the input data is assumed to be in radians.
            Otherwise, the data is assumed to be in degrees.

        Returns
        -------
        (float, float):
            The geodesic area (meters^2) and perimeter (meters) of the polygon.
        """
    return self._polygon_area_perimeter(_copytobuffer(lons)[0], _copytobuffer(lats)[0], radians=radians)