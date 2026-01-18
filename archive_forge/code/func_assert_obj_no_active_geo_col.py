import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def assert_obj_no_active_geo_col(result, expected_type, geo_colname=None):
    """
    Helper method to make tests easier to read. Checks result is of the expected
    type. Asserts that accessing result.geometry.name raises, corresponding to
    _geometry_column_name being in an invalid state
    (either None, or a column no longer present)
    This amounts to testing the assertion raised (geometry column is unset, vs
    old geometry column is missing)

    We assert that _geometry_column_name = int_geo_colname

    """
    if expected_type == GeoDataFrame:
        if geo_colname is None:
            assert result._geometry_column_name is None
        else:
            assert geo_colname == result._geometry_column_name
        if result._geometry_column_name is None:
            msg = 'You are calling a geospatial method on the GeoDataFrame, but the active'
        else:
            msg = f"You are calling a geospatial method on the GeoDataFrame, but the active geometry column \\('{result._geometry_column_name}'\\) is not present"
        with pytest.raises(AttributeError, match=msg):
            result.geometry.name
    else:
        raise NotImplementedError()