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
def computeNodeBreadths(self, graph):
    columns = self.computeNodeColumns(graph)
    _, y0, _, y1 = self.p.bounds
    max_column_size = max(map(len, columns))
    max_default_padding = 20
    py = self.p.node_padding if self.p.node_padding is not None else min((y1 - y0) / (max_column_size - 1), max_default_padding) if max_column_size > 1 else max_default_padding
    self.initializeNodeBreadths(columns, py)
    for i in range(self.p.iterations):
        alpha = 0.99 ** i
        beta = max(1 - alpha, (i + 1) / self.p.iterations)
        self.relaxRightToLeft(columns, alpha, beta, py)
        self.relaxLeftToRight(columns, alpha, beta, py)
    for node in graph['nodes']:
        node['y1'] = round(node['y1'], _Y_N_DECIMAL_DIGITS)