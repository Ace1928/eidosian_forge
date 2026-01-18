import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_no_successors_no_predecessors(self):
    g = graph.DiGraph()
    g.add_node('a')
    g.add_node('b')
    g.add_node('c')
    g.add_edge('b', 'c')
    self.assertEqual(set(['a', 'b']), set(g.no_predecessors_iter()))
    self.assertEqual(set(['a', 'c']), set(g.no_successors_iter()))