from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('data,good,bad,expected, equality', [[[1, 2, 3], 'data.index(2)', 'data.append(4)', 1, True], [{'a': 1}, 'data.keys().isdisjoint({})', 'data.update()', True, True], [CallCreatesHeapType(), 'data()', 'data.__class__()', HeapType, False], [CallCreatesBuiltin(), 'data()', 'data.__class__()', frozenset, False]])
def test_evaluates_calls(data, good, bad, expected, equality):
    context = limited(data=data)
    value = guarded_eval(good, context)
    if equality:
        assert value == expected
    else:
        assert isinstance(value, expected)
    with pytest.raises(GuardRejection):
        guarded_eval(bad, context)