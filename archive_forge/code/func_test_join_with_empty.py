import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('predicate', ['contains', 'contains_properly', 'covered_by', 'covers', 'crosses', 'intersects', 'touches', 'within'])
@pytest.mark.parametrize('empty', [GeoDataFrame(geometry=[GeometryCollection(), GeometryCollection()]), GeoDataFrame(geometry=GeoSeries())])
def test_join_with_empty(self, predicate, empty):
    polygons = geopandas.GeoDataFrame({'col2': [1, 2], 'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]})
    result = sjoin(empty, polygons, how='left', predicate=predicate)
    assert result.index_right.isnull().all()
    result = sjoin(empty, polygons, how='right', predicate=predicate)
    assert result.index_left.isnull().all()
    result = sjoin(empty, polygons, how='inner', predicate=predicate)
    assert result.empty