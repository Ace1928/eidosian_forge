import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_attribute_string_predicate(self):
    t = self._build_tree()
    self.assertEqual(1, len(find_all(t, "//NameNode[@name = 'decorator']")))