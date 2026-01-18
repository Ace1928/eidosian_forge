from matplotlib.path import Path
import numpy as np
import shapely.geometry as sgeom
def path_to_geos(path, force_ccw=False):
    """
    Create a list of Shapely geometric objects from a
    :class:`matplotlib.path.Path`.

    Parameters
    ----------
    path
        A :class:`matplotlib.path.Path` instance.

    Other Parameters
    ----------------
    force_ccw
        Boolean flag determining whether the path can be inverted to enforce
        ccw. Defaults to False.

    Returns
    -------
    A list of instances of the following type(s):
        :class:`shapely.geometry.polygon.Polygon`,
        :class:`shapely.geometry.linestring.LineString` and/or
        :class:`shapely.geometry.multilinestring.MultiLineString`.

    """
    path_verts, path_codes = path_segments(path, curves=False)
    verts_split_inds = np.where(path_codes == Path.MOVETO)[0]
    verts_split = np.split(path_verts, verts_split_inds)
    codes_split = np.split(path_codes, verts_split_inds)
    other_result_geoms = []
    collection = []
    for path_verts, path_codes in zip(verts_split, codes_split):
        if len(path_verts) == 0:
            continue
        if path_codes[-1] == Path.CLOSEPOLY:
            path_verts[-1, :] = path_verts[0, :]
        verts_same_as_first = np.isclose(path_verts[0, :], path_verts[1:, :], rtol=1e-10, atol=1e-13)
        verts_same_as_first = np.logical_and.reduce(verts_same_as_first, axis=1)
        if all(verts_same_as_first):
            geom = sgeom.Point(path_verts[0, :])
        elif path_verts.shape[0] > 4 and path_codes[-1] == Path.CLOSEPOLY:
            geom = sgeom.Polygon(path_verts[:-1, :])
        else:
            geom = sgeom.LineString(path_verts)
        if geom.is_empty:
            pass
        elif len(collection) > 0 and isinstance(collection[-1][0], sgeom.Polygon) and isinstance(geom, sgeom.Polygon) and collection[-1][0].contains(geom.exterior):
            if any((internal.contains(geom) for internal in collection[-1][1])):
                collection.append((geom, []))
            else:
                collection[-1][1].append(geom)
        elif isinstance(geom, sgeom.Point):
            other_result_geoms.append(geom)
        else:
            collection.append((geom, []))
    geom_collection = []
    for external_geom, internal_polys in collection:
        if internal_polys:
            exteriors = [geom.exterior for geom in internal_polys]
            geom = sgeom.Polygon(external_geom.exterior, exteriors)
        else:
            geom = external_geom
        if isinstance(geom, sgeom.Polygon):
            if force_ccw and (not geom.exterior.is_ccw):
                geom = sgeom.polygon.orient(geom)
        geom_collection.append(geom)
    if geom_collection and all((isinstance(geom, sgeom.LineString) for geom in geom_collection)):
        geom_collection = [sgeom.MultiLineString(geom_collection)]

    def not_zero_poly(geom):
        return isinstance(geom, sgeom.Polygon) and (not geom.is_empty) and (geom.area != 0) or not isinstance(geom, sgeom.Polygon)
    result = list(filter(not_zero_poly, geom_collection))
    return result + other_result_geoms