from itertools import combinations
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal
def cheeger(G, k):
    return min((len(nx.node_boundary(G, nn)) / k for nn in combinations(G, k)))