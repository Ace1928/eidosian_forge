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
@test_utils.requires_features('exec_node')
def test_exec_node_default(self):
    src = 'exec foo in bar'
    t = ast.parse(src)
    self.assertEqual('exec(foo, bar)\n', pasta.dump(t))