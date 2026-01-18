import unittest
from traits.observation._observer_graph import ObserverGraph
def test_equality_order_of_children(self):
    graph1 = ObserverGraph(node=1, children=[ObserverGraph(node=2), ObserverGraph(node=3)])
    graph2 = ObserverGraph(node=1, children=[ObserverGraph(node=3), ObserverGraph(node=2)])
    self.assertEqual(graph1, graph2)
    self.assertEqual(hash(graph1), hash(graph2))