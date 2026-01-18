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
def axis_info(self) -> list[Axis]:
    """
        Retrieves all relevant axis information in the CRS.
        If it is a Bound CRS, it gets the axis list from the Source CRS.
        If it is a Compound CRS, it gets the axis list from the Sub CRS list.

        Returns
        -------
        list[Axis]:
            The list of axis information.
        """
    return self._crs.axis_info