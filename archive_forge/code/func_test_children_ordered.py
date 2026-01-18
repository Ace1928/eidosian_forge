import unittest
from traits.observation._observer_graph import ObserverGraph
def test_children_ordered(self):
    child_graph = ObserverGraph(node=2)
    graph = ObserverGraph(node=1, children=[child_graph, ObserverGraph(node=3)])
    self.assertIs(graph.children[0], child_graph)