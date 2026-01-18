from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
def test_callable_args(func, args):
    """
    Return True when this function can be called with the given arguments.
    """
    assert isinstance(args, (list, tuple))
    signature = getattr(inspect, 'signature', None)
    if signature is not None:
        try:
            sig = _signatures_cache[func]
        except KeyError:
            sig = signature(func)
            _signatures_cache[func] = sig
        try:
            sig.bind(*args)
        except TypeError:
            return False
        else:
            return True
    else:
        spec = inspect.getargspec(func)

        def drop_self(spec):
            args, varargs, varkw, defaults = spec
            if args[0:1] == ['self']:
                args = args[1:]
            return inspect.ArgSpec(args, varargs, varkw, defaults)
        spec = drop_self(spec)
        if spec.varargs is not None:
            return True
        return len(spec.args) - len(spec.defaults or []) <= len(args) <= len(spec.args)