import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path(self):
    t = self._build_tree()
    self.assertEqual(2, len(find_all(t, '//DefNode')))
    self.assertEqual(2, len(find_all(t, '//NameNode')))
    self.assertEqual(1, len(find_all(t, '//ReturnStatNode')))
    self.assertEqual(1, len(find_all(t, '//DefNode//ReturnStatNode')))