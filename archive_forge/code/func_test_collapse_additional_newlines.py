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
def test_collapse_additional_newlines(self):
    self.do_layout_test('      # The following multiple newlines should be collapsed into a single newline\n\n\n\n\n      cmake_minimum_required(VERSION 2.8.11)\n      ', [(NodeType.BODY, 0, 0, 0, 75, [(NodeType.COMMENT, 0, 0, 0, 75, []), (NodeType.WHITESPACE, 0, 1, 0, 0, []), (NodeType.STATEMENT, 0, 2, 0, 38, [(NodeType.FUNNAME, 0, 2, 0, 22, []), (NodeType.LPAREN, 0, 2, 22, 23, []), (NodeType.ARGGROUP, 0, 2, 23, 37, [(NodeType.KWARGGROUP, 0, 2, 23, 37, [(NodeType.KEYWORD, 0, 2, 23, 30, []), (NodeType.ARGGROUP, 0, 2, 31, 37, [(NodeType.PARGGROUP, 0, 2, 31, 37, [(NodeType.ARGUMENT, 0, 2, 31, 37, [])])])])]), (NodeType.RPAREN, 0, 2, 37, 38, [])])])])