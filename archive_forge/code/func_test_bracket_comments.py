from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_bracket_comments(self):
    self.assert_tok_types('foo(bar #[=[hello world]=] baz)', [TokenType.WORD, TokenType.LEFT_PAREN, TokenType.WORD, TokenType.WHITESPACE, TokenType.BRACKET_COMMENT, TokenType.WHITESPACE, TokenType.WORD, TokenType.RIGHT_PAREN])
    self.assert_tok_types('      #[==[This is a bracket comment at some nested level\n      #    it is preserved verbatim, but trailing\n      #    whitespace is removed.]==]\n      ', [TokenType.WHITESPACE, TokenType.BRACKET_COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE])
    self.assert_tok_types('      #[[First bracket comment]]\n      # intervening comment\n      #[[Second bracket comment]]\n      ', [TokenType.WHITESPACE, TokenType.BRACKET_COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE, TokenType.BRACKET_COMMENT, TokenType.NEWLINE, TokenType.WHITESPACE])