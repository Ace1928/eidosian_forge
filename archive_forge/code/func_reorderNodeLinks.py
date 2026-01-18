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
def reorderNodeLinks(cls, node):
    for link in node['targetLinks']:
        link['source']['sourceLinks'].sort(key=cmp_to_key(cls.ascendingTargetBreadth))
    for link in node['sourceLinks']:
        link['target']['targetLinks'].sort(key=cmp_to_key(cls.ascendingSourceBreadth))