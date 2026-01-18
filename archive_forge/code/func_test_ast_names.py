import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def test_ast_names():
    test_data = [('np.log(x)', ['np', 'x']), ('x', ['x']), ('center(x + 1)', ['center', 'x']), ('dt.date.dt.month', ['dt'])]
    for code, expected in test_data:
        assert set(ast_names(code)) == set(expected)