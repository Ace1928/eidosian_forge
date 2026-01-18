from matplotlib.path import Path
import numpy as np
import shapely.geometry as sgeom
def geos_to_path(shape):
    """
    Create a list of :class:`matplotlib.path.Path` objects that describe
    a shape.

    Parameters
    ----------
    shape
        A list, tuple or single instance of any of the following
        types: :class:`shapely.geometry.point.Point`,
        :class:`shapely.geometry.linestring.LineString`,
        :class:`shapely.geometry.linestring.LinearRing`,
        :class:`shapely.geometry.polygon.Polygon`,
        :class:`shapely.geometry.multipoint.MultiPoint`,
        :class:`shapely.geometry.multipolygon.MultiPolygon`,
        :class:`shapely.geometry.multilinestring.MultiLineString`,
        :class:`shapely.geometry.collection.GeometryCollection`,
        or any type with a _as_mpl_path() method.

    Returns
    -------
    paths
        A list of :class:`matplotlib.path.Path` objects.

    """
    if isinstance(shape, (list, tuple)):
        paths = []
        for shp in shape:
            paths.extend(geos_to_path(shp))
        return paths
    if isinstance(shape, sgeom.LinearRing):
        return [Path(np.column_stack(shape.xy), closed=True)]
    elif isinstance(shape, (sgeom.LineString, sgeom.Point)):
        return [Path(np.column_stack(shape.xy))]
    elif isinstance(shape, sgeom.Polygon):

        def poly_codes(poly):
            codes = np.ones(len(poly.xy[0])) * Path.LINETO
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            return codes
        if shape.is_empty:
            return []
        vertices = np.concatenate([np.array(shape.exterior.xy)] + [np.array(ring.xy) for ring in shape.interiors], 1).T
        codes = np.concatenate([poly_codes(shape.exterior)] + [poly_codes(ring) for ring in shape.interiors])
        return [Path(vertices, codes)]
    elif isinstance(shape, (sgeom.MultiPolygon, sgeom.GeometryCollection, sgeom.MultiLineString, sgeom.MultiPoint)):
        paths = []
        for geom in shape.geoms:
            paths.extend(geos_to_path(geom))
        return paths
    elif hasattr(shape, '_as_mpl_path'):
        vertices, codes = shape._as_mpl_path()
        return [Path(vertices, codes)]
    else:
        raise ValueError(f'Unsupported shape type {type(shape)}.')