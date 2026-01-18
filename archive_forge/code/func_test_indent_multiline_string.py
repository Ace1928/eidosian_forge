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
def test_indent_multiline_string(self):
    src = textwrap.dedent('        class A:\n          """Doc\n             string."""\n          pass\n        ')
    t = pasta.parse(src)
    docstring, pass_stmt = t.body[0].body
    self.assertEqual('  ', fmt.get(docstring, 'indent'))
    self.assertEqual('  ', fmt.get(pass_stmt, 'indent'))