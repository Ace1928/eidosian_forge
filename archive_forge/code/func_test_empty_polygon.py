from matplotlib.path import Path
import shapely.geometry as sgeom
import cartopy.mpl.patch as cpatch
def test_empty_polygon(self):
    p = Path([[0, 0], [0, 0], [0, 0], [0, 0], [1, 2], [1, 2], [1, 2], [1, 2], [2, 3], [2, 3], [2, 3], [42, 42], [193.75, -14.166664123535156], [193.75, -14.166664123535158], [193.75, -14.166664123535156], [193.75, -14.166664123535156]], codes=[1, 2, 2, 79] * 4)
    geoms = cpatch.path_to_geos(p)
    assert [type(geom) for geom in geoms] == [sgeom.Point] * 4
    assert len(geoms) == 4