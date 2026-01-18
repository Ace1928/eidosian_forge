import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
@pytest.fixture
def merged_shapes(nybb_polydf):
    manhattan_bronx = nybb_polydf.loc[3:4]
    others = nybb_polydf.loc[0:2]
    collapsed = [others.geometry.unary_union, manhattan_bronx.geometry.unary_union]
    merged_shapes = GeoDataFrame({'myshapes': collapsed}, geometry='myshapes', index=pd.Index([5, 6], name='manhattan_bronx'), crs=nybb_polydf.crs)
    return merged_shapes