import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_bidirectional_dijkstra_restricted():
    XG = nx.DiGraph()
    XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
    XG3 = nx.Graph()
    XG3.add_weighted_edges_from([[0, 1, 2], [1, 2, 12], [2, 3, 1], [3, 4, 5], [4, 5, 1], [5, 0, 10]])
    validate_length_path(XG, 's', 'v', 9, *_bidirectional_dijkstra(XG, 's', 'v'))
    validate_length_path(XG, 's', 'v', 10, *_bidirectional_dijkstra(XG, 's', 'v', ignore_nodes=['u']))
    validate_length_path(XG, 's', 'v', 11, *_bidirectional_dijkstra(XG, 's', 'v', ignore_edges=[('s', 'x')]))
    pytest.raises(nx.NetworkXNoPath, _bidirectional_dijkstra, XG, 's', 'v', ignore_nodes=['u'], ignore_edges=[('s', 'x')])
    validate_length_path(XG3, 0, 3, 15, *_bidirectional_dijkstra(XG3, 0, 3))
    validate_length_path(XG3, 0, 3, 16, *_bidirectional_dijkstra(XG3, 0, 3, ignore_nodes=[1]))
    validate_length_path(XG3, 0, 3, 16, *_bidirectional_dijkstra(XG3, 0, 3, ignore_edges=[(2, 3)]))
    pytest.raises(nx.NetworkXNoPath, _bidirectional_dijkstra, XG3, 0, 3, ignore_nodes=[1], ignore_edges=[(5, 4)])