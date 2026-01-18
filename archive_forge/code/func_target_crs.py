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
@property
def target_crs(self) -> Optional['CRS']:
    """
        .. versionadded:: 2.2.0

        Returns
        -------
        CRS:
            The hub CRS of a BoundCRS or the target CRS of a CoordinateOperation.

        """
    return None if self._crs.target_crs is None else CRS(self._crs.target_crs)