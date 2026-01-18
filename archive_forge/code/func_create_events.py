import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def create_events(fname, spec, args, kwds):
    values = dict()
    if spec.defaults:
        values = dict(zip(spec.args[-len(spec.defaults):], spec.defaults))
    values.update(kwds)
    values.update(list(zip(spec.args[:len(args)], args)))
    positional = ['%s=%r' % (a, values.pop(a)) for a in spec.args]
    anonymous = [str(a) for a in args[len(positional):]]
    keywords = ['%s=%r' % (k, values[k]) for k in sorted(values.keys())]
    params = ', '.join([f for f in chain(positional, anonymous, keywords) if f])
    enter = ['>> ', tls.indent * ' ', fname, '(', params, ')']
    leave = ['<< ', tls.indent * ' ', fname]
    return (enter, leave)