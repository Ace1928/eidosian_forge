import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_former_infloop_case(self):
    coords = [(260.625, 68.90383337092122), (360.0, 79.8556091996901), (360.0, 77.76848175458498), (0.0, 88.79068047337279), (210.0, 90.0), (135.0, 88.79068047337279), (260.625, 68.90383337092122)]
    geom = sgeom.Polygon(coords)
    target_projection = ccrs.PlateCarree()
    source_crs = ccrs.Geodetic()
    multi_polygon = target_projection.project_geometry(geom, source_crs)
    assert not multi_polygon.is_empty