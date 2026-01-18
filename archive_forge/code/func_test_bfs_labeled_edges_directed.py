from functools import partial
import pytest
import networkx as nx
def test_bfs_labeled_edges_directed(self):
    D = nx.cycle_graph(5, create_using=nx.DiGraph)
    expected = [(0, 1, 'tree'), (1, 2, 'tree'), (2, 3, 'tree'), (3, 4, 'tree'), (4, 0, 'reverse')]
    answer = list(nx.bfs_labeled_edges(D, 0))
    assert expected == answer
    D.add_edge(4, 4)
    expected.append((4, 4, 'level'))
    answer = list(nx.bfs_labeled_edges(D, 0))
    assert expected == answer
    D.add_edge(0, 2)
    D.add_edge(1, 5)
    D.add_edge(2, 5)
    D.remove_edge(4, 4)
    expected = [(0, 1, 'tree'), (0, 2, 'tree'), (1, 2, 'level'), (1, 5, 'tree'), (2, 3, 'tree'), (2, 5, 'forward'), (3, 4, 'tree'), (4, 0, 'reverse')]
    answer = list(nx.bfs_labeled_edges(D, 0))
    assert expected == answer
    G = D.to_undirected()
    G.add_edge(4, 4)
    expected = [(0, 1, 'tree'), (0, 2, 'tree'), (0, 4, 'tree'), (1, 2, 'level'), (1, 5, 'tree'), (2, 3, 'tree'), (2, 5, 'forward'), (4, 3, 'forward'), (4, 4, 'level')]
    answer = list(nx.bfs_labeled_edges(G, 0))
    assert expected == answer