import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def geom_length(geom):
    from spatialpandas.geometry import MultiLine, MultiPolygon, Polygon, Ring
    if isinstance(geom, Polygon):
        offset = 0
        exterior = geom.data[0]
        if exterior[0] != exterior[-2] or exterior[1] != exterior[-1]:
            offset = 1
        return len(exterior) // 2 + offset
    elif isinstance(geom, (MultiPolygon, MultiLine)):
        length = 0
        for g in geom.data:
            offset = 0
            if isinstance(geom, MultiLine):
                exterior = g
            else:
                exterior = g[0]
                if exterior[0] != exterior[-2] or exterior[1] != exterior[-1]:
                    offset = 1
            length += len(exterior) // 2 + 1 + offset
        return length - 1 if length else 0
    else:
        offset = 0
        exterior = geom.buffer_values
        if isinstance(geom, Ring) and (exterior[0] != exterior[-2] or exterior[1] != exterior[-1]):
            offset = 1
        return len(exterior) // 2