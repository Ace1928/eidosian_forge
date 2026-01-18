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
def test_annotated_tokens():
    tokens_without_origins = [(token_type, token, props) for token_type, token, origin, props in annotated_tokens('a(b) + c.d')]
    assert tokens_without_origins == [(tokenize.NAME, 'a', {'bare_ref': True, 'bare_funcall': True}), (tokenize.OP, '(', {'bare_ref': False, 'bare_funcall': False}), (tokenize.NAME, 'b', {'bare_ref': True, 'bare_funcall': False}), (tokenize.OP, ')', {'bare_ref': False, 'bare_funcall': False}), (tokenize.OP, '+', {'bare_ref': False, 'bare_funcall': False}), (tokenize.NAME, 'c', {'bare_ref': True, 'bare_funcall': False}), (tokenize.OP, '.', {'bare_ref': False, 'bare_funcall': False}), (tokenize.NAME, 'd', {'bare_ref': False, 'bare_funcall': False})]
    assert len(list(annotated_tokens('x'))) == 1