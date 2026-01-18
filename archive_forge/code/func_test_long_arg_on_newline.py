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
def test_long_arg_on_newline(self):
    self.do_layout_test('      # This command has a very long argument and can\'t be aligned with the command\n      # end, so it should be moved to a new line with block indent + 1.\n      some_long_command_name("Some very long argument that really needs to be on the next line.")\n      ', [(NodeType.BODY, 0, 0, 0, 77, [(NodeType.COMMENT, 0, 0, 0, 77, []), (NodeType.STATEMENT, 1, 2, 0, 70, [(NodeType.FUNNAME, 0, 2, 0, 22, []), (NodeType.LPAREN, 0, 2, 22, 23, []), (NodeType.ARGGROUP, 0, 3, 2, 69, [(NodeType.PARGGROUP, 0, 3, 2, 69, [(NodeType.ARGUMENT, 0, 3, 2, 69, [])])]), (NodeType.RPAREN, 0, 3, 69, 70, [])])])])