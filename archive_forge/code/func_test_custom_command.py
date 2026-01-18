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
def test_custom_command(self):
    self.do_layout_test('      # This very long command should be broken up along keyword arguments\n      foo(nonkwarg_a nonkwarg_b HEADERS a.h b.h c.h d.h e.h f.h SOURCES a.cc b.cc d.cc DEPENDS foo bar baz)\n      ', [(NodeType.BODY, 0, 0, 0, 68, [(NodeType.COMMENT, 0, 0, 0, 68, []), (NodeType.STATEMENT, 4, 1, 0, 35, [(NodeType.FUNNAME, 0, 1, 0, 3, []), (NodeType.LPAREN, 0, 1, 3, 4, []), (NodeType.ARGGROUP, 4, 1, 4, 35, [(NodeType.PARGGROUP, 0, 1, 4, 25, [(NodeType.ARGUMENT, 0, 1, 4, 14, []), (NodeType.ARGUMENT, 0, 1, 15, 25, [])]), (NodeType.KWARGGROUP, 0, 2, 4, 35, [(NodeType.KEYWORD, 0, 2, 4, 11, []), (NodeType.ARGGROUP, 0, 2, 12, 35, [(NodeType.PARGGROUP, 0, 2, 12, 35, [(NodeType.ARGUMENT, 0, 2, 12, 15, []), (NodeType.ARGUMENT, 0, 2, 16, 19, []), (NodeType.ARGUMENT, 0, 2, 20, 23, []), (NodeType.ARGUMENT, 0, 2, 24, 27, []), (NodeType.ARGUMENT, 0, 2, 28, 31, []), (NodeType.ARGUMENT, 0, 2, 32, 35, [])])])]), (NodeType.KWARGGROUP, 0, 3, 4, 26, [(NodeType.KEYWORD, 0, 3, 4, 11, []), (NodeType.ARGGROUP, 0, 3, 12, 26, [(NodeType.PARGGROUP, 0, 3, 12, 26, [(NodeType.ARGUMENT, 0, 3, 12, 16, []), (NodeType.ARGUMENT, 0, 3, 17, 21, []), (NodeType.ARGUMENT, 0, 3, 22, 26, [])])])]), (NodeType.KWARGGROUP, 0, 4, 4, 15, [(NodeType.KEYWORD, 0, 4, 4, 11, []), (NodeType.ARGGROUP, 0, 4, 12, 15, [(NodeType.PARGGROUP, 0, 4, 12, 15, [(NodeType.ARGUMENT, 0, 4, 12, 15, [])])])]), (NodeType.PARGGROUP, 0, 5, 4, 11, [(NodeType.FLAG, 0, 5, 4, 7, []), (NodeType.FLAG, 0, 5, 8, 11, [])])]), (NodeType.RPAREN, 0, 5, 11, 12, [])])])])