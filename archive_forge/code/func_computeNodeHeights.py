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
def computeNodeHeights(cls, graph):
    nodes = graph['nodes']
    height = 0
    while nodes:
        next_nodes = []
        for node in nodes:
            node['height'] = height
            for link in node['targetLinks']:
                next_nodes.append(link['source'])
        nodes = next_nodes
        height += 1
        if height > len(graph['nodes']):
            raise RecursionError('Sankey diagrams only support acyclic graphs.')
    return height