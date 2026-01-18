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
def test_EvalFactor_end_to_end():
    from patsy.state import stateful_transform
    foo = stateful_transform(_MockTransform)
    e = EvalFactor('foo(x) + foo(foo(y))')
    state = {}
    eval_env = EvalEnvironment.capture(0)
    passes = e.memorize_passes_needed(state, eval_env)
    print(passes)
    print(state)
    assert passes == 2
    assert state['eval_env'].namespace['foo'] is foo
    for name in ['x', 'y', 'e', 'state']:
        assert name not in state['eval_env'].namespace
    import numpy as np
    e.memorize_chunk(state, 0, {'x': np.array([1, 2]), 'y': np.array([10, 11])})
    assert state['transforms']['_patsy_stobj0__foo__']._memorize_chunk_called == 1
    assert state['transforms']['_patsy_stobj2__foo__']._memorize_chunk_called == 1
    e.memorize_chunk(state, 0, {'x': np.array([12, -10]), 'y': np.array([100, 3])})
    assert state['transforms']['_patsy_stobj0__foo__']._memorize_chunk_called == 2
    assert state['transforms']['_patsy_stobj2__foo__']._memorize_chunk_called == 2
    assert state['transforms']['_patsy_stobj0__foo__']._memorize_finish_called == 0
    assert state['transforms']['_patsy_stobj2__foo__']._memorize_finish_called == 0
    e.memorize_finish(state, 0)
    assert state['transforms']['_patsy_stobj0__foo__']._memorize_finish_called == 1
    assert state['transforms']['_patsy_stobj2__foo__']._memorize_finish_called == 1
    assert state['transforms']['_patsy_stobj1__foo__']._memorize_chunk_called == 0
    assert state['transforms']['_patsy_stobj1__foo__']._memorize_finish_called == 0
    e.memorize_chunk(state, 1, {'x': np.array([1, 2]), 'y': np.array([10, 11])})
    e.memorize_chunk(state, 1, {'x': np.array([12, -10]), 'y': np.array([100, 3])})
    e.memorize_finish(state, 1)
    for transform in six.itervalues(state['transforms']):
        assert transform._memorize_chunk_called == 2
        assert transform._memorize_finish_called == 1
    assert np.all(e.eval(state, {'x': np.array([1, 2, 12, -10]), 'y': np.array([10, 11, 100, 3])}) == [254, 256, 355, 236])