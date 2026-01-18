import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def edge_ins_cost(attr):
    if attr['color'] == 'blue':
        return 0.4
    else:
        return 1.0