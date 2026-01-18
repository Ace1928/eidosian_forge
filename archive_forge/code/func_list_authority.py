import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
def list_authority(self, auth_name: Optional[str]=None, min_confidence: int=70) -> list[AuthorityMatchInfo]:
    """
        .. versionadded:: 3.2.0

        Return the authority names and codes best matching the CRS.

        Example:

        >>> from pyproj import CRS
        >>> ccs = CRS("EPSG:4328")
        >>> ccs.list_authority()
        [AuthorityMatchInfo(auth_name='EPSG', code='4326', confidence=100)]

        If the CRS is bound, you can get an authority from
        the source CRS:

        >>> from pyproj import CRS
        >>> ccs = CRS("+proj=geocent +datum=WGS84 +towgs84=0,0,0")
        >>> ccs.list_authority()
        []
        >>> ccs.source_crs.list_authority()
        [AuthorityMatchInfo(auth_name='EPSG', code='4978', confidence=70)]
        >>> ccs == CRS.from_authorty('EPSG', '4978')
        False

        Parameters
        ----------
        auth_name: str, optional
            The name of the authority to filter by.
        min_confidence: int, default=70
            A value between 0-100 where 100 is the most confident.
            :ref:`min_confidence`

        Returns
        -------
        list[AuthorityMatchInfo]:
            List of authority matches for the CRS.
        """
    return self._crs.list_authority(auth_name=auth_name, min_confidence=min_confidence)