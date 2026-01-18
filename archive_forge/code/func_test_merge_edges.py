import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
def test_merge_edges(self):
    g = graph.DiGraph()
    g.add_node('a')
    g.add_node('b')
    g.add_edge('a', 'b')
    g2 = graph.DiGraph()
    g2.add_node('c')
    g2.add_node('d')
    g2.add_edge('c', 'd')
    g3 = graph.merge_graphs(g, g2)
    self.assertEqual(4, len(g3))
    self.assertTrue(g3.has_edge('c', 'd'))
    self.assertTrue(g3.has_edge('a', 'b'))