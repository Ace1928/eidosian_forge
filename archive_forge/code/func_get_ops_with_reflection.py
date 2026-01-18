from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def get_ops_with_reflection():
    """
    Return operators with a reflection, along with their partial derivatives.

    Operators are things like +, /, etc. Those considered here have two
    arguments and can be called through Python's reflected methods __r…__ (e.g.
    __radd__).

    See the code for details.
    """
    derivatives_list = {'add': ('1.', '1.'), 'div': ('1/y', '-x/y**2'), 'floordiv': ('0.', '0.'), 'mod': ('1.', 'partial_derivative(float.__mod__, 1)(x, y)'), 'mul': ('y', 'x'), 'sub': ('1.', '-1.'), 'truediv': ('1/y', '-x/y**2')}
    ops_with_reflection = {}
    for op, derivatives in derivatives_list.items():
        ops_with_reflection[op] = [eval('lambda x, y: %s' % expr) for expr in derivatives]
        ops_with_reflection['r' + op] = [eval('lambda y, x: %s' % expr) for expr in reversed(derivatives)]

    def pow_deriv_0(x, y):
        if y == 0:
            return 0.0
        elif x != 0 or y % 1 == 0:
            return y * x ** (y - 1)
        else:
            return float('nan')

    def pow_deriv_1(x, y):
        if x == 0 and y > 0:
            return 0.0
        else:
            return log(x) * x ** y
    ops_with_reflection['pow'] = [pow_deriv_0, pow_deriv_1]
    ops_with_reflection['rpow'] = [lambda y, x: pow_deriv_1(x, y), lambda y, x: pow_deriv_0(x, y)]
    for op in ['pow']:
        ops_with_reflection[op] = [nan_if_exception(func) for func in ops_with_reflection[op]]
        ops_with_reflection['r' + op] = [nan_if_exception(func) for func in ops_with_reflection['r' + op]]
    return ops_with_reflection