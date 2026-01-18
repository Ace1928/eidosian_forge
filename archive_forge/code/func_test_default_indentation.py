from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import os.path
import unittest
from six import with_metaclass
import pasta
from pasta.base import codegen
from pasta.base import test_utils
def test_default_indentation(self):
    for indent in ('  ', '    ', '\t'):
        src = 'def a():\n' + indent + 'b\n'
        t = pasta.parse(src)
        t.body.extend(ast.parse('def c(): d').body)
        self.assertEqual(codegen.to_str(t), src + 'def c():\n' + indent + 'd\n')