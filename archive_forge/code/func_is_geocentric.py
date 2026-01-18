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
def is_geocentric(self) -> bool:
    """
        This checks if the CRS is geocentric and
        takes into account if the CRS is bound.

        Returns
        -------
        bool:
            True if CRS is in geocentric (x/y) coordinates.
        """
    return self._crs.is_geocentric