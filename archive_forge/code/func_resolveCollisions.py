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
def resolveCollisions(self, nodes, alpha, py):
    _, y0, _, y1 = self.p.bounds
    i = len(nodes) // 2
    subject = nodes[i]
    self.resolveCollisionsBottomToTop(nodes, subject['y0'] - py, i - 1, alpha, py)
    self.resolveCollisionsTopToBottom(nodes, subject['y1'] + py, i + 1, alpha, py)
    self.resolveCollisionsBottomToTop(nodes, y1, len(nodes) - 1, alpha, py)
    self.resolveCollisionsTopToBottom(nodes, y0, 0, alpha, py)