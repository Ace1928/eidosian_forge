import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def repr_pretty_impl(p, obj, args, kwargs=[]):
    name = obj.__class__.__name__
    p.begin_group(len(name) + 1, '%s(' % (name,))
    started = [False]

    def new_item():
        if started[0]:
            p.text(',')
            p.breakable()
        started[0] = True
    for arg in args:
        new_item()
        p.pretty(arg)
    for label, value in kwargs:
        new_item()
        p.begin_group(len(label) + 1, '%s=' % (label,))
        p.pretty(value)
        p.end_group(len(label) + 1, '')
    p.end_group(len(name) + 1, ')')