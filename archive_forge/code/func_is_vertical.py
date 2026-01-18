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
def is_vertical(self) -> bool:
    """
        .. versionadded:: 2.2.0

        This checks if the CRS is vertical.
        It will check if it has a vertical CRS
        in the sub CRS if it is a compound CRS and will check if
        the source CRS is vertical if it is a bound CRS.

        Returns
        -------
        bool:
            True if CRS is vertical.
        """
    return self._crs.is_vertical