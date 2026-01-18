from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_byteorder_mark(self):
    self.assert_tok_types('\ufeffcmake_minimum_required(VERSION 2.8.11)', [TokenType.BYTEORDER_MARK, TokenType.WORD, TokenType.LEFT_PAREN, TokenType.WORD, TokenType.WHITESPACE, TokenType.UNQUOTED_LITERAL, TokenType.RIGHT_PAREN])