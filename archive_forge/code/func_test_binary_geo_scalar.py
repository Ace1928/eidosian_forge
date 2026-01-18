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
@pytest.mark.parametrize('attr', ['difference', 'symmetric_difference', 'union', 'intersection'])
def test_binary_geo_scalar(attr):
    na_value = None
    quads = []
    while len(quads) < 1:
        geom = shapely.geometry.Polygon([(random.random(), random.random()) for i in range(4)])
        if geom.is_valid:
            quads.append(geom)
    q = quads[0]
    for other in [q, shapely.geometry.Polygon()]:
        result = getattr(T, attr)(other)
        expected = [getattr(t, attr)(other) if t is not None else na_value for t in triangles]
    assert equal_geometries(result, expected)