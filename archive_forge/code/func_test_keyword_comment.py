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
def test_keyword_comment(self):
    self.do_layout_test('      find_package(package REQUIRED\n                   COMPONENTS # --------------------------------------\n                              # @TODO: This has to be filled manually\n                              # --------------------------------------\n                              this_is_a_really_long_word_foo)\n      ', [(NodeType.BODY, 0, 0, 0, 53, [(NodeType.STATEMENT, 4, 0, 0, 53, [(NodeType.FUNNAME, 0, 0, 0, 12, []), (NodeType.LPAREN, 0, 0, 12, 13, []), (NodeType.ARGGROUP, 4, 1, 2, 53, [(NodeType.PARGGROUP, 0, 1, 2, 18, [(NodeType.ARGUMENT, 0, 1, 2, 9, []), (NodeType.FLAG, 0, 1, 10, 18, [])]), (NodeType.KWARGGROUP, 4, 2, 2, 53, [(NodeType.KEYWORD, 0, 2, 2, 12, []), (NodeType.ARGGROUP, 4, 2, 13, 53, [(NodeType.COMMENT, 0, 2, 13, 53, []), (NodeType.PARGGROUP, 0, 5, 13, 43, [(NodeType.ARGUMENT, 0, 5, 13, 43, [])])])])]), (NodeType.RPAREN, 0, 5, 43, 44, [])])])])