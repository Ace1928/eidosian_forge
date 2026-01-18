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
def area_of_use(self) -> Optional[AreaOfUse]:
    """
        Returns
        -------
        AreaOfUse:
            The area of use object with associated attributes.
        """
    return self._crs.area_of_use