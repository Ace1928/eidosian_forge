from collections import defaultdict
from types import FunctionType
import numpy as np
import pandas as pd
import param
from ..core import Dataset, Dimension, Element2D
from ..core.accessors import Redim
from ..core.operation import Operation
from ..core.util import is_dataframe, max_range, search_indices
from .chart import Points
from .path import Path
from .util import (
@classmethod
def from_vertices(cls, data):
    """
        Uses Delauney triangulation to compute triangle simplices for
        each point.
        """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError('Generating triangles from points requires SciPy to be installed.') from None
    if not isinstance(data, Points):
        data = Points(data)
    if not len(data):
        return cls(([], []))
    tris = Delaunay(data.array([0, 1]))
    return cls((tris.simplices, data))