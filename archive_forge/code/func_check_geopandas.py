from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_point import geom_point
from .geom_polygon import geom_polygon
def check_geopandas():
    try:
        import geopandas
    except ImportError as e:
        msg = 'geom_map requires geopandas. Please install geopandas.'
        raise PlotnineError(msg) from e