import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_cuts(self):
    linear_ring = sgeom.LinearRing([(-10, 30), (10, 60), (10, 50)])
    projection = ccrs.Robinson(170.5)
    rings, multi_line_string = projection.project_geometry(linear_ring)
    assert len(multi_line_string.geoms) > 1
    assert not rings

    def assert_close_to_boundary(xy):
        limit = (projection.x_limits[1] - projection.x_limits[0]) / 10000.0
        assert sgeom.Point(*xy).distance(projection.boundary) < limit, 'Bad topology near boundary'
    for line_string in multi_line_string.geoms:
        coords = list(line_string.coords)
        assert len(coords) >= 2
        assert_close_to_boundary(coords[0])
        assert_close_to_boundary(coords[-1])