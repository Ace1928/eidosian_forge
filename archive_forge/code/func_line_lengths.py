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
def line_lengths(self, lons: Any, lats: Any, radians: bool=False) -> Any:
    """
        .. versionadded:: 2.3.0

        Calculate the distances between points along a line (meters).

        >>> from pyproj import Geod
        >>> geod = Geod(ellps="WGS84")
        >>> lats = [-72.9, -71.9, -74.9]
        >>> lons = [-74, -102, -102]
        >>> for line_length in geod.line_lengths(lons, lats):
        ...     f"{line_length:.3f}"
        '943065.744'
        '334805.010'

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
        array, :class:`numpy.ndarray`, list, tuple, or scalar:
            The total length of the line (meters).
        """
    inx, x_data_type = _copytobuffer(lons)
    iny = _copytobuffer(lats)[0]
    self._line_length(inx, iny, radians=radians)
    line_lengths = _convertback(x_data_type, inx)
    return line_lengths if x_data_type == DataType.FLOAT else line_lengths[:-1]