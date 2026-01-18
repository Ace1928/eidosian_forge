import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
def test_metric_closure(self):
    M = metric_closure(self.G1)
    mc = [(1, 2, {'distance': 10, 'path': [1, 2]}), (1, 3, {'distance': 20, 'path': [1, 2, 3]}), (1, 4, {'distance': 22, 'path': [1, 2, 7, 5, 4]}), (1, 5, {'distance': 12, 'path': [1, 2, 7, 5]}), (1, 6, {'distance': 22, 'path': [1, 2, 7, 5, 6]}), (1, 7, {'distance': 11, 'path': [1, 2, 7]}), (2, 3, {'distance': 10, 'path': [2, 3]}), (2, 4, {'distance': 12, 'path': [2, 7, 5, 4]}), (2, 5, {'distance': 2, 'path': [2, 7, 5]}), (2, 6, {'distance': 12, 'path': [2, 7, 5, 6]}), (2, 7, {'distance': 1, 'path': [2, 7]}), (3, 4, {'distance': 10, 'path': [3, 4]}), (3, 5, {'distance': 12, 'path': [3, 2, 7, 5]}), (3, 6, {'distance': 22, 'path': [3, 2, 7, 5, 6]}), (3, 7, {'distance': 11, 'path': [3, 2, 7]}), (4, 5, {'distance': 10, 'path': [4, 5]}), (4, 6, {'distance': 20, 'path': [4, 5, 6]}), (4, 7, {'distance': 11, 'path': [4, 5, 7]}), (5, 6, {'distance': 10, 'path': [5, 6]}), (5, 7, {'distance': 1, 'path': [5, 7]}), (6, 7, {'distance': 11, 'path': [6, 5, 7]})]
    assert edges_equal(list(M.edges(data=True)), mc)