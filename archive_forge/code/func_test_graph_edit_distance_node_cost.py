import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_node_cost(self):
    G1 = path_graph(6)
    G2 = path_graph(6)
    for n, attr in G1.nodes.items():
        attr['color'] = 'red' if n % 2 == 0 else 'blue'
    for n, attr in G2.nodes.items():
        attr['color'] = 'red' if n % 2 == 1 else 'blue'

    def node_subst_cost(uattr, vattr):
        if uattr['color'] == vattr['color']:
            return 1
        else:
            return 10

    def node_del_cost(attr):
        if attr['color'] == 'blue':
            return 20
        else:
            return 50

    def node_ins_cost(attr):
        if attr['color'] == 'blue':
            return 40
        else:
            return 100
    assert graph_edit_distance(G1, G2, node_subst_cost=node_subst_cost, node_del_cost=node_del_cost, node_ins_cost=node_ins_cost) == 6