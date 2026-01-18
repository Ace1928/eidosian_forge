import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_attribute_exists(self):
    t = self._build_tree()
    self.assertEqual(2, len(find_all(t, '//NameNode[@name]')))
    self.assertEqual(ExprNodes.NameNode, type(find_first(t, '//NameNode[@name]')))