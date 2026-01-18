import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_3pt_poly(self):
    projection = ccrs.OSGB(approx=True)
    polygon = sgeom.Polygon([(-1000, -1000), (-1000, 200000), (200000, -1000)])
    multi_polygon = projection.project_geometry(polygon, ccrs.OSGB(approx=True))
    assert len(multi_polygon.geoms) == 1
    assert len(multi_polygon.geoms[0].exterior.coords) == 4