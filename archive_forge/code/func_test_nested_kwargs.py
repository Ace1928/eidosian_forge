from __future__ import unicode_literals
import unittest
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.printer import tree_string, test_string
from cmakelang.parse.common import NodeType
def test_nested_kwargs(self):
    self.do_type_test('      add_custom_target(name ALL VERBATIM\n        COMMAND echo hello world\n        COMMENT "this is some text")\n      ', [(NodeType.BODY, [(NodeType.WHITESPACE, []), (NodeType.STATEMENT, [(NodeType.FUNNAME, []), (NodeType.LPAREN, []), (NodeType.ARGGROUP, [(NodeType.PARGGROUP, [(NodeType.ARGUMENT, []), (NodeType.FLAG, [])]), (NodeType.PARGGROUP, [(NodeType.FLAG, [])]), (NodeType.KWARGGROUP, [(NodeType.KEYWORD, []), (NodeType.ARGGROUP, [(NodeType.PARGGROUP, [(NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, [])])])]), (NodeType.KWARGGROUP, [(NodeType.KEYWORD, []), (NodeType.PARGGROUP, [(NodeType.ARGUMENT, [])])])]), (NodeType.RPAREN, [])]), (NodeType.WHITESPACE, [])])])