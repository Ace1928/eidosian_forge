import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_inverted_poly_simple_hole(self):
    proj = ccrs.NorthPolarStereo()
    poly = sgeom.Polygon([(0, 0), (-90, 0), (-180, 0), (-270, 0)], [[(0, -30), (90, -30), (180, -30), (270, -30)]])
    multi_polygon = proj.project_geometry(poly)
    assert len(multi_polygon.geoms) == 1
    assert len(multi_polygon.geoms[0].interiors) == 1
    polygon = multi_polygon.geoms[0]
    self._assert_bounds(polygon.bounds, -24000000.0, -24000000.0, 24000000.0, 24000000.0, 1000000.0)
    self._assert_bounds(polygon.interiors[0].bounds, -12000000.0, -12000000.0, 12000000.0, 12000000.0, 1000000.0)