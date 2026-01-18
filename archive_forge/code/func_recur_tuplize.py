from llvmlite import ir
from llvmlite import binding as ll
from numba.core import datamodel
import unittest
def recur_tuplize(args, func=None):
    for arg in args:
        if isinstance(arg, (tuple, list)):
            yield tuple(recur_tuplize(arg, func=func))
        elif func is None:
            yield arg
        else:
            yield func(arg)