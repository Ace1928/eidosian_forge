from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_atword(self):
    self.assert_tok_types('@PACKAGE_INIT@', [TokenType.ATWORD])