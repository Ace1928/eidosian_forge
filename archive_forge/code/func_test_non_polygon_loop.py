from matplotlib.path import Path
import shapely.geometry as sgeom
import cartopy.mpl.patch as cpatch
def test_non_polygon_loop(self):
    p = Path([[0, 10], [170, 20], [-170, 30], [0, 10]], codes=[1, 2, 2, 2])
    geoms = cpatch.path_to_geos(p)
    assert [type(geom) for geom in geoms] == [sgeom.MultiLineString]
    assert len(geoms) == 1