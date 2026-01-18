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
def relaxLeftToRight(self, columns, alpha, beta, py):
    for column in columns[1:]:
        for target in column:
            y = 0
            w = 0
            for link in target['targetLinks']:
                source = link['source']
                v = link['value'] * (target['column'] - source['column'])
                y += self.targetTop(source, target, py) * v
                w += v
            if w <= 0:
                continue
            dy = (y / w - target['y0']) * alpha
            target['y0'] += dy
            target['y1'] += dy
            self.reorderNodeLinks(target)
        if self.p.node_sort:
            column.sort(key=cmp_to_key(self.ascendingBreadth))
        self.resolveCollisions(column, beta, py)