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
def from_user_input(cls, value: Any, **kwargs) -> 'CRS':
    """
        Initialize a CRS class instance with:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

        Parameters
        ----------
        value : obj
            A Python int, dict, or str.

        Returns
        -------
        CRS
        """
    if isinstance(value, cls):
        return value
    crs = cls.__new__(cls)
    super(CustomConstructorCRS, crs).__init__(value, **kwargs)
    crs._check_type()
    return crs