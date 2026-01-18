from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_import_reads_in_functiondef(self):
    source = textwrap.dedent('        import aaa\n        @aaa.x\n        def foo(bar):\n          return aaa\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    return_value = nodes[1].body[0].value
    decorator = nodes[1].decorator_list[0].value
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [decorator, return_value])