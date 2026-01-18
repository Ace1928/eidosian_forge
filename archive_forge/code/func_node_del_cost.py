import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def node_del_cost(attr):
    if attr['color'] == 'blue':
        return 20
    else:
        return 50