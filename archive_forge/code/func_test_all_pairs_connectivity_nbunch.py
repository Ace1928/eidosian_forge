import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_all_pairs_connectivity_nbunch(self):
    G = nx.complete_graph(5)
    nbunch = [0, 2, 3]
    C = nx.all_pairs_node_connectivity(G, nbunch=nbunch)
    assert len(C) == len(nbunch)