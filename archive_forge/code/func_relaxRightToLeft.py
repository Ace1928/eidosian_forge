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
def relaxRightToLeft(self, columns, alpha, beta, py):
    """Reposition each node based on its outgoing (source) links."""
    for column in columns[-2::-1]:
        for source in column:
            y = 0
            w = 0
            for link in source['sourceLinks']:
                target = link['target']
                v = link['value'] * (target['column'] - source['column'])
                y += self.sourceTop(source, target, py) * v
                w += v
            if w <= 0:
                continue
            dy = (y / w - source['y0']) * alpha
            source['y0'] += dy
            source['y1'] += dy
            self.reorderNodeLinks(source)
        if self.p.node_sort:
            column.sort(key=cmp_to_key(self.ascendingBreadth))
        self.resolveCollisions(column, beta, py)