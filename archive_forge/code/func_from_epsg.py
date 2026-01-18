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
@classmethod
def from_epsg(cls, code: Union[str, int]) -> 'CRS':
    """Make a CRS from an EPSG code

        Parameters
        ----------
        code : int or str
            An EPSG code.

        Returns
        -------
        CRS
        """
    return cls.from_user_input(_prepare_from_epsg(code))