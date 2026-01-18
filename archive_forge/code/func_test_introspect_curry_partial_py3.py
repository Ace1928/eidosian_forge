import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_introspect_curry_partial_py3():
    test_introspect_curry_valid_py3(check_valid=is_partial_args, incomplete=True)