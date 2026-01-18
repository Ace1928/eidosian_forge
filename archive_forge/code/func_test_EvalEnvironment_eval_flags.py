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
def test_EvalEnvironment_eval_flags():
    import pytest
    if sys.version_info >= (3,):
        test_flag = __future__.barry_as_FLUFL.compiler_flag
        assert test_flag & _ALL_FUTURE_FLAGS
        env = EvalEnvironment([{'a': 11}], flags=0)
        assert env.eval('a != 0') == True
        pytest.raises(SyntaxError, env.eval, 'a <> 0')
        assert env.subset(['a']).flags == 0
        assert env.with_outer_namespace({'b': 10}).flags == 0
        env2 = EvalEnvironment([{'a': 11}], flags=test_flag)
        assert env2.eval('a <> 0') == True
        pytest.raises(SyntaxError, env2.eval, 'a != 0')
        assert env2.subset(['a']).flags == test_flag
        assert env2.with_outer_namespace({'b': 10}).flags == test_flag
    else:
        test_flag = __future__.division.compiler_flag
        assert test_flag & _ALL_FUTURE_FLAGS
        env = EvalEnvironment([{'a': 11}], flags=0)
        assert env.eval('a / 2') == 11 // 2 == 5
        assert env.subset(['a']).flags == 0
        assert env.with_outer_namespace({'b': 10}).flags == 0
        env2 = EvalEnvironment([{'a': 11}], flags=test_flag)
        assert env2.eval('a / 2') == 11 * 1.0 / 2 != 5
        env2.subset(['a']).flags == test_flag
        assert env2.with_outer_namespace({'b': 10}).flags == test_flag