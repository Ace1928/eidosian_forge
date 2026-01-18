import random
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry
import shapely.wkb
import shapely.wkt
import geopandas
from geopandas.array import (
import geopandas._compat as compat
import pytest
def test_geom_types_null_mixed():
    geoms = [shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1)]), None, shapely.geometry.Point(0, 1)]
    G = from_shapely(geoms)
    cat = G.geom_type
    assert list(cat) == ['Polygon', None, 'Point']