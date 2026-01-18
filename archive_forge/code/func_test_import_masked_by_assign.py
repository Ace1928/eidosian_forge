from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_import_masked_by_assign(self):
    source = textwrap.dedent('        import aaa\n        def foo():\n          aaa = 123\n          return aaa\n        aaa\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_aaa = nodes[2].value
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [node_aaa])