import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_at_boundary(self):
    exterior = np.array([[177.5, -79.912], [178.333, -79.946], [181.666, -83.494], [180.833, -83.57], [180.0, -83.62], [178.438, -83.333], [178.333, -83.312], [177.956, -83.888], [180.0, -84.086], [180.833, -84.318], [183.0, -86.0], [183.0, -78.0], [177.5, -79.912]])
    tring = sgeom.LinearRing(exterior)
    tcrs = ccrs.PlateCarree()
    scrs = ccrs.PlateCarree()
    rings, mlinestr = tcrs._project_linear_ring(tring, scrs)
    assert len(mlinestr.geoms) == 4
    assert not rings
    assert round(abs(mlinestr.convex_hull.area - 2347.7562), 4) == 0