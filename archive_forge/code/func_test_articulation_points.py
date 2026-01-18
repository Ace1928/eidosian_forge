import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_articulation_points():
    Ggen = _generate_no_biconnected()
    for i in range(1):
        G = next(Ggen)
        articulation_points = [{a} for a in nx.articulation_points(G)]
        for cut in nx.all_node_cuts(G):
            assert cut in articulation_points