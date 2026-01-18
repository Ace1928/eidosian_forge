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
def test_call_no_pos(self):
    """Tests that Call node traversal works without position information."""
    src = 'f(a)'
    t = pasta.parse(src)
    node = ast_utils.find_nodes_by_type(t, (ast.Call,))[0]
    node.keywords.append(ast.keyword(arg='b', value=ast.Num(n=0)))
    self.assertEqual('f(a, b=0)', pasta.dump(t))