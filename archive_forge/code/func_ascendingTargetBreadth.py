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
def ascendingTargetBreadth(cls, a, b):
    return (cls.ascendingBreadth(a['target'], b['target']) if 'y0' in a['target'] and 'y0' in b['target'] else None) or a['index'] - b['index']