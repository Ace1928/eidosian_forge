from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def test__tokenize_formula():
    code = 'y ~ a + (foo(b,c +   2)) + -1 + 0 + 10'
    tokens = list(_tokenize_formula(code, ['+', '-', '~']))
    expecteds = [('PYTHON_EXPR', Origin(code, 0, 1), 'y'), ('~', Origin(code, 2, 3), None), ('PYTHON_EXPR', Origin(code, 4, 5), 'a'), ('+', Origin(code, 6, 7), None), (Token.LPAREN, Origin(code, 8, 9), None), ('PYTHON_EXPR', Origin(code, 9, 23), 'foo(b, c + 2)'), (Token.RPAREN, Origin(code, 23, 24), None), ('+', Origin(code, 25, 26), None), ('-', Origin(code, 27, 28), None), ('ONE', Origin(code, 28, 29), '1'), ('+', Origin(code, 30, 31), None), ('ZERO', Origin(code, 32, 33), '0'), ('+', Origin(code, 34, 35), None), ('NUMBER', Origin(code, 36, 38), '10')]
    for got, expected in zip(tokens, expecteds):
        assert isinstance(got, Token)
        assert got.type == expected[0]
        assert got.origin == expected[1]
        assert got.extra == expected[2]