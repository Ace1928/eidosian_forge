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
def test_argcomment_preserved_and_reflowed(self):
    self.do_layout_test('      set(HEADERS header_a.h header_b.h # This comment should\n                                        # be preserved, moreover it should be split\n                                        # across two lines.\n          header_c.h header_d.h)\n      ', [(NodeType.BODY, 0, 0, 0, 80, [(NodeType.STATEMENT, 4, 0, 0, 80, [(NodeType.FUNNAME, 0, 0, 0, 3, []), (NodeType.LPAREN, 0, 0, 3, 4, []), (NodeType.ARGGROUP, 4, 0, 4, 80, [(NodeType.PARGGROUP, 0, 0, 4, 11, [(NodeType.ARGUMENT, 0, 0, 4, 11, [])]), (NodeType.PARGGROUP, 0, 1, 4, 80, [(NodeType.ARGUMENT, 0, 1, 4, 14, []), (NodeType.ARGUMENT, 0, 1, 15, 80, [(NodeType.COMMENT, 0, 1, 26, 80, [])]), (NodeType.ARGUMENT, 0, 3, 4, 14, []), (NodeType.ARGUMENT, 0, 3, 15, 25, [])])]), (NodeType.RPAREN, 0, 3, 25, 26, [])])])])