import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
@staticmethod
def from_proj(proj_from: Any, proj_to: Any, always_xy: bool=False, area_of_interest: Optional[AreaOfInterest]=None) -> 'Transformer':
    """Make a Transformer from a :obj:`pyproj.Proj` or input used to create one.

        .. deprecated:: 3.4.1 :meth:`~Transformer.from_crs` is preferred.

        .. versionadded:: 2.2.0 always_xy
        .. versionadded:: 2.3.0 area_of_interest

        Parameters
        ----------
        proj_from: :obj:`pyproj.Proj` or input used to create one
            Projection of input data.
        proj_to: :obj:`pyproj.Proj` or input used to create one
            Projection of output data.
        always_xy: bool, default=False
            If true, the transform method will accept as input and return as output
            coordinates using the traditional GIS order, that is longitude, latitude
            for geographic CRS and easting, northing for most projected CRS.
        area_of_interest: :class:`.AreaOfInterest`, optional
            The area of interest to help select the transformation.

        Returns
        -------
        Transformer

        """
    from pyproj import Proj
    if not isinstance(proj_from, Proj):
        proj_from = Proj(proj_from)
    if not isinstance(proj_to, Proj):
        proj_to = Proj(proj_to)
    return Transformer.from_crs(proj_from.crs, proj_to.crs, always_xy=always_xy, area_of_interest=area_of_interest)