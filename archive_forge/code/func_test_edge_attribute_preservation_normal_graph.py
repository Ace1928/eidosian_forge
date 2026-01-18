import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_edge_attribute_preservation_normal_graph():
    G = nx.Graph()
    edgelist = [(0, 1, [('weight', 5), ('otherattr', 1), ('otherattr2', 3)]), (0, 2, [('weight', 5), ('otherattr', 2), ('otherattr2', 2)]), (1, 2, [('weight', 6), ('otherattr', 3), ('otherattr2', 1)])]
    G.add_edges_from(edgelist)
    B = branchings.maximum_branching(G, preserve_attrs=True)
    assert B[0][1]['otherattr'] == 1
    assert B[0][1]['otherattr2'] == 3