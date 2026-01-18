from __future__ import absolute_import, division, print_function
import unittest
from datashader import datashape
from datashader.datashape import lexer
def test_failing_tokens(self):
    self.check_failing_token('~')
    self.check_failing_token('`')
    self.check_failing_token('@')
    self.check_failing_token('$')
    self.check_failing_token('%')
    self.check_failing_token('^')
    self.check_failing_token('&')
    self.check_failing_token('-')
    self.check_failing_token('+')
    self.check_failing_token(';')
    self.check_failing_token('<')
    self.check_failing_token('>')
    self.check_failing_token('.')
    self.check_failing_token('..')
    self.check_failing_token('/')
    self.check_failing_token('|')
    self.check_failing_token('\\')