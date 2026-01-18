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
def add_operators_to_AffineScalarFunc():
    """
    Adds many operators (__add__, etc.) to the AffineScalarFunc class.
    """

    def _simple_add_deriv(x):
        if x >= 0:
            return 1.0
        else:
            return -1.0
    simple_numerical_operators_derivatives = {'abs': _simple_add_deriv, 'neg': lambda x: -1.0, 'pos': lambda x: 1.0, 'trunc': lambda x: 0.0}
    for op, derivative in iter(simple_numerical_operators_derivatives.items()):
        attribute_name = '__%s__' % op
        try:
            setattr(AffineScalarFunc, attribute_name, wrap(getattr(float, attribute_name), [derivative]))
        except AttributeError:
            pass
        else:
            modified_operators.append(op)
    for op, derivatives in ops_with_reflection.items():
        attribute_name = '__%s__' % op
        try:
            if op not in custom_ops:
                func_to_wrap = getattr(float, attribute_name)
            else:
                func_to_wrap = custom_ops[op]
        except AttributeError:
            pass
        else:
            setattr(AffineScalarFunc, attribute_name, wrap(func_to_wrap, derivatives))
            modified_ops_with_reflection.append(op)
    for coercion_type in ('complex', 'int', 'long', 'float'):

        def raise_error(self):
            raise TypeError("can't convert an affine function (%s) to %s; use x.nominal_value" % (self.__class__, coercion_type))
        setattr(AffineScalarFunc, '__%s__' % coercion_type, raise_error)