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
def capture_obj_method_calls(obj_name, code):
    capturers = []
    for token_type, token, origin, props in annotated_tokens(code):
        for capturer in capturers:
            capturer.add_token(token_type, token)
        if props['bare_ref'] and token == obj_name:
            capturers.append(_FuncallCapturer(token_type, token))
    return [(''.join(capturer.func), pretty_untokenize(capturer.tokens)) for capturer in capturers]