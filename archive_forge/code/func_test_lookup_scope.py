from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_lookup_scope(self):
    src = textwrap.dedent('        import a\n        def b(c, d, e=1):\n          class F(d):\n            g = 1\n          return c\n        ')
    t = ast.parse(src)
    import_node, func_node = t.body
    class_node, return_node = func_node.body
    sc = scope.analyze(t)
    import_node_scope = sc.lookup_scope(import_node)
    self.assertIs(import_node_scope.node, t)
    self.assertIs(import_node_scope, sc)
    self.assertItemsEqual(import_node_scope.names, ['a', 'b'])
    func_node_scope = sc.lookup_scope(func_node)
    self.assertIs(func_node_scope.node, func_node)
    self.assertIs(func_node_scope.parent_scope, sc)
    self.assertItemsEqual(func_node_scope.names, ['c', 'd', 'e', 'F'])
    class_node_scope = sc.lookup_scope(class_node)
    self.assertIs(class_node_scope.node, class_node)
    self.assertIs(class_node_scope.parent_scope, func_node_scope)
    self.assertItemsEqual(class_node_scope.names, ['g'])
    return_node_scope = sc.lookup_scope(return_node)
    self.assertIs(return_node_scope.node, func_node)
    self.assertIs(return_node_scope, func_node_scope)
    self.assertItemsEqual(return_node_scope.names, ['c', 'd', 'e', 'F'])
    self.assertIs(class_node_scope.lookup_scope(func_node), func_node_scope)
    self.assertIsNone(sc.lookup_scope(ast.Name(id='foo')))