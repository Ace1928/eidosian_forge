import datetime
import inspect
import re
import statistics
from functools import wraps
from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get
def null_if_any(*required):
    """
    Decorator that makes a function return `None` if any of the `required` arguments are `None`.

    This also supports decoration with no arguments, e.g.:

        @null_if_any
        def foo(a, b): ...

    In which case all arguments are required.
    """
    f = None
    if len(required) == 1 and callable(required[0]):
        f = required[0]
        required = ()

    def decorator(func):
        if required:
            required_indices = [i for i, param in enumerate(inspect.signature(func).parameters) if param in required]

            def predicate(*args):
                return any((args[i] is None for i in required_indices))
        else:

            def predicate(*args):
                return any((a is None for a in args))

        @wraps(func)
        def _func(*args):
            if predicate(*args):
                return None
            return func(*args)
        return _func
    if f:
        return decorator(f)
    return decorator