import unittest
from traits.observation._observer_graph import ObserverGraph
def test_children_unique(self):
    child_graph = ObserverGraph(node=2)
    with self.assertRaises(ValueError) as exception_cm:
        ObserverGraph(node=1, children=[child_graph, ObserverGraph(node=2)])
    self.assertEqual(str(exception_cm.exception), 'Not all children are unique.')