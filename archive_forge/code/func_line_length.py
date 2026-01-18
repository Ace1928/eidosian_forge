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
def line_length(self, lons: Any, lats: Any, radians: bool=False) -> float:
    """
        .. versionadded:: 2.3.0

        Calculate the total distance between points along a line (meters).

        >>> from pyproj import Geod
        >>> geod = Geod('+a=6378137 +f=0.0033528106647475126')
        >>> lats = [-72.9, -71.9, -74.9, -74.3, -77.5, -77.4, -71.7, -65.9, -65.7,
        ...         -66.6, -66.9, -69.8, -70.0, -71.0, -77.3, -77.9, -74.7]
        >>> lons = [-74, -102, -102, -131, -163, 163, 172, 140, 113,
        ...         88, 59, 25, -4, -14, -33, -46, -61]
        >>> total_length = geod.line_length(lons, lats)
        >>> f"{total_length:.3f}"
        '14259605.611'


        Parameters
        ----------
        lons: array, :class:`numpy.ndarray`, list, tuple, or scalar
            The longitude points along a line.
        lats: array, :class:`numpy.ndarray`, list, tuple, or scalar
            The latitude points along a line.
        radians: bool, default=False
            If True, the input data is assumed to be in radians.
            Otherwise, the data is assumed to be in degrees.

        Returns
        -------
        float:
            The total length of the line (meters).
        """
    inx = _copytobuffer(lons)[0]
    iny = _copytobuffer(lats)[0]
    return self._line_length(inx, iny, radians=radians)