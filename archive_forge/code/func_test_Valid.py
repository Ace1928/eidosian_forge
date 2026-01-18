import gyp.common
import unittest
import sys
def test_Valid(self):
    """Test that sorting works on a valid graph with one possible order."""
    graph = {'a': ['b', 'c'], 'b': [], 'c': ['d'], 'd': ['b']}

    def GetEdge(node):
        return tuple(graph[node])
    self.assertEqual(gyp.common.TopologicallySorted(graph.keys(), GetEdge), ['a', 'c', 'd', 'b'])