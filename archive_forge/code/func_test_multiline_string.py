from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def test_multiline_string(self):
    self.do_layout_test('      foo(some_arg some_arg "\n          This string is on multiple lines\n      ")\n      ', [(NodeType.BODY, 0, 0, 0, 36, [(NodeType.STATEMENT, 0, 0, 0, 36, [(NodeType.FUNNAME, 0, 0, 0, 3, []), (NodeType.LPAREN, 0, 0, 3, 4, []), (NodeType.ARGGROUP, 0, 0, 4, 36, [(NodeType.PARGGROUP, 0, 0, 4, 36, [(NodeType.ARGUMENT, 0, 0, 4, 12, []), (NodeType.ARGUMENT, 0, 0, 13, 21, []), (NodeType.ARGUMENT, 0, 0, 22, 36, [])])]), (NodeType.RPAREN, 0, 2, 1, 2, [])])])])