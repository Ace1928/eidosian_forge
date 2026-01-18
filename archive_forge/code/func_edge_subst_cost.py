import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def edge_subst_cost(gattr, hattr):
    if gattr['color'] == hattr['color']:
        return 0.01
    else:
        return 0.1