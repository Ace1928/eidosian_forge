import unittest
from traits.observation._observer_graph import ObserverGraph
def test_equality_different_type(self):
    graph1 = graph_from_nodes(1, 2, 3)
    self.assertNotEqual(graph1, 1)