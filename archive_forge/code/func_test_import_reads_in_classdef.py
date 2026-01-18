from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_import_reads_in_classdef(self):
    source = textwrap.dedent('        import aaa\n        @aaa.x\n        class Foo(aaa.Bar):\n          pass\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_aaa = nodes[0].names[0]
    decorator = nodes[1].decorator_list[0].value
    base = nodes[1].bases[0].value
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'Foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [decorator, base])