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
def test_EvalEnvironment_eq():
    env1 = EvalEnvironment.capture(0)
    env2 = EvalEnvironment.capture(0)
    assert env1 == env2
    assert hash(env1) == hash(env2)
    capture_local_env = lambda: EvalEnvironment.capture(0)
    env3 = capture_local_env()
    env4 = capture_local_env()
    assert env3 != env4