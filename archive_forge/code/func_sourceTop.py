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
def sourceTop(cls, source, target, py):
    y = target['y0'] - (len(target['targetLinks']) - 1) * py / 2
    for link in target['targetLinks']:
        if link['source'] is source:
            break
        y += link['width'] + py
    for link in source['sourceLinks']:
        if link['target'] is target:
            break
        y -= link['width']
    return y