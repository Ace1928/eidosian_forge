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
def prime_meridian(self) -> Optional[PrimeMeridian]:
    """
        .. versionadded:: 2.2.0

        Returns
        -------
        PrimeMeridian:
            The prime meridian object with associated attributes.
        """
    return self._crs.prime_meridian