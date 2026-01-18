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
@property
def ccw_boundary(self):
    try:
        boundary = self._ccw_boundary
    except AttributeError:
        boundary = sgeom.LinearRing(self.boundary.coords[::-1])
        self._ccw_boundary = boundary
    return boundary