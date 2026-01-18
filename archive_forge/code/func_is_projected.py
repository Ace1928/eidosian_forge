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
def is_projected(self) -> bool:
    """
        This checks if the CRS is projected.
        It will check if it has a projected CRS
        in the sub CRS if it is a compound CRS and will check if
        the source CRS is projected if it is a bound CRS.

        Returns
        -------
        bool:
            True if CRS is projected.
        """
    return self._crs.is_projected