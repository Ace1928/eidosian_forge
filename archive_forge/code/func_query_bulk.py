import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
@doc(BaseSpatialIndex.query_bulk)
def query_bulk(self, geometry, predicate=None, sort=False):
    warnings.warn('The `query_bulk()` method is deprecated and will be removed in GeoPandas 1.0. You can use the `query()` method instead.', FutureWarning, stacklevel=2)
    return self.query(geometry, predicate=predicate, sort=sort)