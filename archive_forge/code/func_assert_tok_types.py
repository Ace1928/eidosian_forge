from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def assert_tok_types(self, input_str, expected_types):
    """
    Run the lexer on the input string and assert that the result tokens match
    the expected
    """
    self.assertEqual(expected_types, [tok.type for tok in lex.tokenize(input_str)])