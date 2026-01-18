from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def epsg(code):
    """
    Return the projection which corresponds to the given EPSG code.

    The EPSG code must correspond to a "projected coordinate system",
    so EPSG codes such as 4326 (WGS-84) which define a "geodetic coordinate
    system" will not work.

    Note
    ----
        The conversion is performed by pyproj.CRS.

    """
    import cartopy._epsg
    return cartopy._epsg._EPSGProjection(code)