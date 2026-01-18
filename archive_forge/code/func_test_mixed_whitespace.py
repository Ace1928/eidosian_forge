from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_mixed_whitespace(self):
    """
    Ensure that if a newline is part of a whitespace sequence then it is
    tokenized separately.
    """
    self.assert_tok_types(' \n', [TokenType.WHITESPACE, TokenType.NEWLINE])
    self.assert_tok_types('\t\n', [TokenType.WHITESPACE, TokenType.NEWLINE])
    self.assert_tok_types('\x0c\n', [TokenType.WHITESPACE, TokenType.NEWLINE])
    self.assert_tok_types('\x0b\n', [TokenType.WHITESPACE, TokenType.NEWLINE])
    self.assert_tok_types('\r\n', [TokenType.NEWLINE])