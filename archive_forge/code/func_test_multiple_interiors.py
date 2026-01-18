import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_multiple_interiors(self):
    exterior = ring(0, 0, 12, 12, True)
    interiors = [ring(1, 1, 2, 2, False), ring(1, 8, 2, 9, False)]
    poly = sgeom.Polygon(exterior, interiors)
    target = ccrs.PlateCarree()
    source = ccrs.Geodetic()
    assert len(list(target.project_geometry(poly, source).geoms)) == 1