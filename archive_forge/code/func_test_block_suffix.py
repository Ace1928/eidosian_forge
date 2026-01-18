from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
def test_block_suffix(self):
    src_tpl = textwrap.dedent('        {open_block}\n          pass #a\n          #b\n            #c\n\n          #d\n        #e\n        a\n        ')
    test_cases = (('body', 'def x():'), ('body', 'class X:'), ('body', 'if x:'), ('orelse', 'if x:\n  y\nelse:'), ('body', 'if x:\n  y\nelif y:'), ('body', 'while x:'), ('orelse', 'while x:\n  y\nelse:'), ('finalbody', 'try:\n  x\nfinally:'), ('body', 'try:\n  x\nexcept:'), ('orelse', 'try:\n  x\nexcept:\n  y\nelse:'), ('body', 'with x:'), ('body', 'with x, y:'), ('body', 'with x:\n with y:'), ('body', 'for x in y:'))

    def is_node_for_suffix(node, children_attr):
        val = getattr(node, children_attr, None)
        return isinstance(val, list) and type(val[0]) == ast.Pass
    for children_attr, open_block in test_cases:
        src = src_tpl.format(open_block=open_block)
        t = pasta.parse(src)
        node_finder = ast_utils.FindNodeVisitor(lambda node: is_node_for_suffix(node, children_attr))
        node_finder.visit(t)
        node = node_finder.results[0]
        expected = '  #b\n    #c\n\n  #d\n'
        actual = str(fmt.get(node, 'block_suffix_%s' % children_attr))
        self.assertMultiLineEqual(expected, actual, 'Incorrect suffix for code:\n%s\nNode: %s (line %d)\nDiff:\n%s' % (src, node, node.lineno, '\n'.join(_get_diff(actual, expected))))
        self.assertMultiLineEqual(src, pasta.dump(t))