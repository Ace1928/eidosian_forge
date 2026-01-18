from matplotlib.path import Path
import shapely.geometry as sgeom
import cartopy.mpl.patch as cpatch
def test_nested_polygons(self):
    vertices = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0], [2, 2], [2, 8], [8, 8], [8, 2], [2, 2], [4, 4], [4, 6], [6, 6], [6, 4], [4, 4]]
    codes = [1, 2, 2, 2, 79, 1, 2, 2, 2, 79, 1, 2, 2, 2, 79]
    p = Path(vertices, codes=codes)
    geoms = cpatch.path_to_geos(p)
    assert len(geoms) == 2
    assert all((isinstance(geom, sgeom.Polygon) for geom in geoms))
    assert len(geoms[0].interiors) == 1
    assert len(geoms[1].interiors) == 0