import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_coord_slice_points(self):
    assert self.df2.cx[-2:-1, -2:-1].empty
    assert_frame_equal(self.df2, self.df2.cx[:, :])
    assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, :])
    assert_frame_equal(self.df2.loc[5:], self.df2.cx[:, 5:])
    assert_frame_equal(self.df2.loc[5:], self.df2.cx[5:, 5:])