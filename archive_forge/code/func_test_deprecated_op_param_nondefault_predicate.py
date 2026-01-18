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
@pytest.mark.parametrize('dfs', ['default-index'], indirect=True)
@pytest.mark.parametrize('op', ['intersects', 'contains', 'within'])
@pytest.mark.parametrize('predicate', ['contains', 'within'])
def test_deprecated_op_param_nondefault_predicate(self, dfs, op, predicate):
    _, df1, df2, _ = dfs
    match = 'use the `predicate` parameter instead'
    if op != predicate:
        warntype = UserWarning
        match = '`predicate` will be overridden by the value of `op`' + '(.|\\s)*' + match
    else:
        warntype = FutureWarning
    with pytest.warns(warntype, match=match):
        sjoin(df1, df2, predicate=predicate, op=op)