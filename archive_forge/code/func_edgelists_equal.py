import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def edgelists_equal(self, e1, e2):
    return sorted((sorted(e) for e in e1)) == sorted((sorted(e) for e in e2))