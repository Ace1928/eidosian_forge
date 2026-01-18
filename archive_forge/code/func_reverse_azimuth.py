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
def reverse_azimuth(azi: Any, radians: bool=False) -> Any:
    """
    Reverses the given azimuth (forward <-> backwards)

    .. versionadded:: 3.5.0

    Accepted numeric scalar or array:

    - :class:`int`
    - :class:`float`
    - :class:`numpy.floating`
    - :class:`numpy.integer`
    - :class:`list`
    - :class:`tuple`
    - :class:`array.array`
    - :class:`numpy.ndarray`
    - :class:`xarray.DataArray`
    - :class:`pandas.Series`

    Parameters
    ----------
    azi: scalar or array
        The azimuth.
    radians: bool, default=False
        If True, the input data is assumed to be in radians.
        Otherwise, the data is assumed to be in degrees.

    Returns
    -------
    scalar or array:
        The reversed azimuth (forward <-> backwards)
    """
    inazi, azi_data_type = _copytobuffer(azi)
    _reverse_azimuth(inazi, radians=radians)
    return _convertback(azi_data_type, inazi)