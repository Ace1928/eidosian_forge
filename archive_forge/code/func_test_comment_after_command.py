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
def test_comment_after_command(self):
    with self.subTest(sub=1):
        self.do_layout_test('        foo_command() # comment\n        ', [(NodeType.BODY, 0, 0, 0, 23, [(NodeType.STATEMENT, 0, 0, 0, 23, [(NodeType.FUNNAME, 0, 0, 0, 11, []), (NodeType.LPAREN, 0, 0, 11, 12, []), (NodeType.ARGGROUP, 0, 0, 12, 12, []), (NodeType.RPAREN, 0, 0, 12, 13, []), (NodeType.COMMENT, 0, 0, 14, 23, [])])])])
    with self.subTest(sub=2):
        self.do_layout_test('        foo_command() # this is a long comment that exceeds the desired page width and will be wrapped to a newline\n        ', [(NodeType.BODY, 0, 0, 0, 78, [(NodeType.STATEMENT, 0, 0, 0, 78, [(NodeType.FUNNAME, 0, 0, 0, 11, []), (NodeType.LPAREN, 0, 0, 11, 12, []), (NodeType.ARGGROUP, 0, 0, 12, 12, []), (NodeType.RPAREN, 0, 0, 12, 13, []), (NodeType.COMMENT, 0, 0, 14, 78, [])])])])