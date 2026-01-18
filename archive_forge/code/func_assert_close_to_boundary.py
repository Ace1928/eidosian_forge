import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def assert_close_to_boundary(xy):
    limit = (projection.x_limits[1] - projection.x_limits[0]) / 10000.0
    assert sgeom.Point(*xy).distance(projection.boundary) < limit, 'Bad topology near boundary'