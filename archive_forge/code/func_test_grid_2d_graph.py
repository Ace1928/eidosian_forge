import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_grid_2d_graph():
    G = nx.grid_2d_graph(5, 5)
    solution = [{(0, 1), (1, 0)}, {(3, 0), (4, 1)}, {(3, 4), (4, 3)}, {(0, 3), (1, 4)}]
    for cut in nx.all_node_cuts(G):
        assert cut in solution