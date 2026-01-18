import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_inspect_wrapped_property():

    class Wrapped(object):

        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

        @property
        def __wrapped__(self):
            return self.func
    func = lambda x: x
    wrapped = Wrapped(func)
    assert inspect.signature(func) == inspect.signature(wrapped)
    assert num_required_args(Wrapped) is None
    _sigs.signatures[Wrapped] = (_sigs.expand_sig((0, lambda func: None)),)
    assert num_required_args(Wrapped) == 1