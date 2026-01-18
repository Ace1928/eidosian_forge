import sys
from collections import OrderedDict
import numpy as np
from holoviews.core.data import Interface, DictInterface, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.core.data.spatialpandas import to_geom_dict
from holoviews.core.dimension import dimension_name
from holoviews.core.util import isscalar
from ..util import asarray, geom_types, geom_to_array, geom_length
def geom_from_dict(geom, xdim, ydim, single_type, multi_type):
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, MultiLineString
    if (xdim, ydim) in geom:
        xs, ys = asarray(geom.pop((xdim, ydim))).T
    elif xdim in geom and ydim in geom:
        xs, ys = (geom.pop(xdim), geom.pop(ydim))
    else:
        raise ValueError('Could not find geometry dimensions')
    xscalar, yscalar = (isscalar(xs), isscalar(ys))
    if xscalar and yscalar:
        xs, ys = (np.array([xs]), np.array([ys]))
    elif xscalar:
        xs = np.full_like(ys, xs)
    elif yscalar:
        ys = np.full_like(xs, ys)
    geom_array = np.column_stack([xs, ys])
    splits = np.where(np.isnan(geom_array[:, :2].astype('float')).sum(axis=1))[0]
    if len(splits):
        split_geoms = [g[:-1] if i == len(splits) - 1 else g for i, g in enumerate(np.split(geom_array, splits + 1))]
    else:
        split_geoms = [geom_array]
    split_holes = geom.pop('holes', None)
    if split_holes is not None and len(split_holes) != len(split_geoms):
        raise DataError('Polygons with holes containing multi-geometries must declare a list of holes for each geometry.')
    if single_type is Point:
        if len(splits) > 1 or any((len(g) > 1 for g in split_geoms)):
            geom = MultiPoint(np.concatenate(split_geoms))
        else:
            geom = Point(*split_geoms[0])
    elif len(splits):
        if multi_type is MultiPolygon:
            if split_holes is None:
                split_holes = [[]] * len(split_geoms)
            geom = MultiPolygon(list(zip(split_geoms, split_holes)))
        else:
            geom = MultiLineString(split_geoms)
    elif single_type is Polygon:
        if split_holes is None or not len(split_holes):
            split_holes = [None]
        geom = Polygon(split_geoms[0], split_holes[0])
    else:
        geom = LineString(split_geoms[0])
    return geom