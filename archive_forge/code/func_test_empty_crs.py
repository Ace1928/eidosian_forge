import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
@pytest.mark.parametrize('driver,ext', driver_ext_pairs)
def test_empty_crs(tmpdir, driver, ext, engine):
    """Test handling of undefined CRS with GPKG driver (GH #1975)."""
    if ext == '.gpkg':
        pytest.xfail('GPKG is read with Undefined geographic SRS.')
    tempfilename = os.path.join(str(tmpdir), 'boros' + ext)
    df = GeoDataFrame({'a': [1.0, 2.0, 3.0], 'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]})
    df.to_file(tempfilename, driver=driver, engine=engine)
    result = read_file(tempfilename, engine=engine)
    if ext == '.geojson':
        df.crs = 'EPSG:4326'
    assert_geodataframe_equal(result, df)