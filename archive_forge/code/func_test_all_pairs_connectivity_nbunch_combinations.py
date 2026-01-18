import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_all_pairs_connectivity_nbunch_combinations(self):
    G = nx.complete_graph(5)
    nbunch = [0, 2, 3]
    A = {n: {} for n in nbunch}
    for u, v in itertools.combinations(nbunch, 2):
        A[u][v] = A[v][u] = nx.node_connectivity(G, u, v)
    C = nx.all_pairs_node_connectivity(G, nbunch=nbunch)
    assert sorted(((k, sorted(v)) for k, v in A.items())) == sorted(((k, sorted(v)) for k, v in C.items()))