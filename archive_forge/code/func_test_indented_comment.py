from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_indented_comment(self):
    self.assert_tok_types('      # This multiline-comment should be reflowed\n      # into a single comment\n      # on one line\n      ', [TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE])