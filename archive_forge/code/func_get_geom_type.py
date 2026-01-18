import sys
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from holoviews.core.util import isscalar, unique_iterator, unique_array
from holoviews.core.data import Dataset, Interface, MultiInterface, PandasAPI
from holoviews.core.data.interface import DataError
from holoviews.core.data import PandasInterface
from holoviews.core.data.spatialpandas import get_value_array
from holoviews.core.dimension import dimension_name
from holoviews.element import Path
from ..util import asarray, geom_to_array, geom_types, geom_length
from .geom_dict import geom_from_dict
def get_geom_type(geom):
    """Returns the HoloViews geometry type.

    Args:
        geom: A shapely geometry

    Returns:
        A string representing type of the geometry.
    """
    from shapely.geometry import Point, LineString, Polygon, Ring, MultiPoint, MultiPolygon, MultiLineString
    if isinstance(geom, (Point, MultiPoint)):
        return 'Point'
    elif isinstance(geom, (LineString, MultiLineString)):
        return 'Line'
    elif isinstance(geom, Ring):
        return 'Ring'
    elif isinstance(geom, (Polygon, MultiPolygon)):
        return 'Polygon'