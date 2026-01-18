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
def auto_format_test_generator(input_file):

    def test(self):
        with open(input_file, 'r') as handle:
            src = handle.read()
        t = ast.parse(src)
        auto_formatted = codegen.to_str(t)
        self.assertMultiLineEqual(src, auto_formatted)
    return test