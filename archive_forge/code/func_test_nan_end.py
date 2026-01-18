import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_nan_end(self):
    projection = ccrs.TransverseMercator(central_longitude=-90, approx=False)
    line_string = sgeom.LineString([(-10, 30), (10, 50)])
    multi_line_string = projection.project_geometry(line_string)
    assert len(multi_line_string.geoms) == 1
    for line_string in multi_line_string.geoms:
        for coord in line_string.coords:
            assert not any(np.isnan(coord)), 'Unexpected NaN in projected coords.'