import sys
from collections import OrderedDict
import numpy as np
from holoviews.core.data import Interface, DictInterface, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.core.data.spatialpandas import to_geom_dict
from holoviews.core.dimension import dimension_name
from holoviews.core.util import isscalar
from ..util import asarray, geom_types, geom_to_array, geom_length
@classmethod
def has_holes(cls, dataset):
    from shapely.geometry import Polygon, MultiPolygon
    geom = dataset.data['geometry']
    if isinstance(geom, Polygon) and geom.interiors:
        return True
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if isinstance(g, Polygon) and g.interiors:
                return True
    return False