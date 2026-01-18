import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_edge_cost(self):
    G1 = path_graph(6)
    G2 = path_graph(6)
    for e, attr in G1.edges.items():
        attr['color'] = 'red' if min(e) % 2 == 0 else 'blue'
    for e, attr in G2.edges.items():
        attr['color'] = 'red' if min(e) // 3 == 0 else 'blue'

    def edge_subst_cost(gattr, hattr):
        if gattr['color'] == hattr['color']:
            return 0.01
        else:
            return 0.1

    def edge_del_cost(attr):
        if attr['color'] == 'blue':
            return 0.2
        else:
            return 0.5

    def edge_ins_cost(attr):
        if attr['color'] == 'blue':
            return 0.4
        else:
            return 1.0
    assert graph_edit_distance(G1, G2, edge_subst_cost=edge_subst_cost, edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost) == 0.23