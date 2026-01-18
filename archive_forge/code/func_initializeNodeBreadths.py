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
def initializeNodeBreadths(self, columns, py):
    _, y0, _, y1 = self.p.bounds
    ky = min(((y1 - y0 - (len(c) - 1) * py) / sum((node['value'] for node in c)) for c in columns))
    for nodes in columns:
        y = y0
        for node in nodes:
            node['y0'] = y
            node['y1'] = y + node['value'] * ky
            y = node['y1'] + py
            for link in node['sourceLinks']:
                link['width'] = link['value'] * ky
        y = (y1 - y + py) / (len(nodes) + 1)
        for i, node in enumerate(nodes):
            node['y0'] += y * (i + 1)
            node['y1'] += y * (i + 1)
        self.reorderLinks(nodes)