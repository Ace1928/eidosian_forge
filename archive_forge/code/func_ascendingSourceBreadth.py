import math
from functools import cmp_to_key
from itertools import cycle
import numpy as np
import param
from ..core.data import Dataset
from ..core.dimension import Dimension
from ..core.operation import Operation
from ..core.util import get_param_values, unique_array
from .graphs import EdgePaths, Graph, Nodes
from .util import quadratic_bezier
@classmethod
def ascendingSourceBreadth(cls, a, b):
    return (cls.ascendingBreadth(a['source'], b['source']) if 'y0' in a['source'] and 'y0' in b['source'] else None) or a['index'] - b['index']