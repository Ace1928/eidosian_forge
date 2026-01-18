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
def from_json_dict(cls, crs_dict: dict) -> 'CRS':
    """
        .. versionadded:: 2.4.0

        Create CRS from a JSON dictionary.

        Parameters
        ----------
        crs_dict: dict
            CRS dictionary.

        Returns
        -------
        CRS
        """
    return cls.from_user_input(json.dumps(crs_dict))