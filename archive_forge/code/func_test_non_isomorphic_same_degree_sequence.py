import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_non_isomorphic_same_degree_sequence(self):
    """
                G1                           G2
        x--------------x              x--------------x
        | \\            |              | \\            |
        |  x-------x   |              |  x-------x   |
        |  |       |   |              |  |       |   |
        |  x-------x   |              |  x-------x   |
        | /            |              |            \\ |
        x--------------x              x--------------x
        """
    edges1 = [(1, 5), (1, 2), (4, 1), (3, 2), (3, 4), (4, 8), (5, 8), (6, 5), (6, 7), (7, 8)]
    edges2 = [(1, 5), (1, 2), (4, 1), (3, 2), (4, 3), (5, 8), (6, 5), (6, 7), (3, 7), (8, 7)]
    G1 = nx.DiGraph(edges1)
    G2 = nx.DiGraph(edges2)
    assert vf2pp_isomorphism(G1, G2) is None