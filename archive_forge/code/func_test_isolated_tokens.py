from __future__ import absolute_import, division, print_function
import unittest
from datashader import datashape
from datashader.datashape import lexer
def test_isolated_tokens(self):
    self.check_isolated_token('testing', 'NAME_LOWER', 'testing')
    self.check_isolated_token('Testing', 'NAME_UPPER', 'Testing')
    self.check_isolated_token('_testing', 'NAME_OTHER', '_testing')
    self.check_isolated_token('*', 'ASTERISK')
    self.check_isolated_token(',', 'COMMA')
    self.check_isolated_token('=', 'EQUAL')
    self.check_isolated_token(':', 'COLON')
    self.check_isolated_token('[', 'LBRACKET')
    self.check_isolated_token(']', 'RBRACKET')
    self.check_isolated_token('{', 'LBRACE')
    self.check_isolated_token('}', 'RBRACE')
    self.check_isolated_token('(', 'LPAREN')
    self.check_isolated_token(')', 'RPAREN')
    self.check_isolated_token('...', 'ELLIPSIS')
    self.check_isolated_token('->', 'RARROW')
    self.check_isolated_token('?', 'QUESTIONMARK')
    self.check_isolated_token('32102', 'INTEGER', 32102)
    self.check_isolated_token('->', 'RARROW')
    self.check_isolated_token('"testing"', 'STRING', 'testing')
    self.check_isolated_token("'testing'", 'STRING', 'testing')