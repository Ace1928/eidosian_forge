from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_complicated_string_with_quotes(self):
    self.assert_tok_types('\n      install(CODE "message(\\"foo ${bar}/${baz}...\\")\n        subfun(COMMAND ${WHEEL_COMMAND}\n                       ERROR_MESSAGE \\"error ${bar}/${baz}\\"\n               )"\n      )\n      ', [TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.WORD, TokenType.LEFT_PAREN, TokenType.WORD, TokenType.WHITESPACE, TokenType.QUOTED_LITERAL, TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.RIGHT_PAREN, TokenType.NEWLINE, TokenType.WHITESPACE])