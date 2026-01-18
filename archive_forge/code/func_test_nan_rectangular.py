import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_nan_rectangular(self):
    projection = ccrs.Robinson()
    line_string = sgeom.LineString([(0, 0), (1, 1), (np.nan, np.nan), (2, 2), (3, 3)])
    multi_line_string = projection.project_geometry(line_string, ccrs.PlateCarree())
    assert len(multi_line_string.geoms) == 2