import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_and(self):
    t = self._build_tree()
    self.assertEqual(1, len(find_all(t, '//DefNode[.//ReturnStatNode and .//NameNode]')))
    self.assertEqual(0, len(find_all(t, '//NameNode[@honking and @name]')))
    self.assertEqual(0, len(find_all(t, '//NameNode[@name and @honking]')))
    self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode[@name] and @name]')))