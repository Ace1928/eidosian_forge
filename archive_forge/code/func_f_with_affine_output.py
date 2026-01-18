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
@set_doc('    Version of %s(...) that returns an affine approximation\n    (AffineScalarFunc object), if its result depends on variables\n    (Variable objects).  Otherwise, returns a simple constant (when\n    applied to constant arguments).\n\n    Warning: arguments of the function that are not AffineScalarFunc\n    objects must not depend on uncertainties.Variable objects in any\n    way.  Otherwise, the dependence of the result in\n    uncertainties.Variable objects will be incorrect.\n\n    Original documentation:\n    %s' % (f.__name__, f.__doc__))
def f_with_affine_output(*args, **kwargs):
    pos_w_uncert = [index for index, value in enumerate(args) if isinstance(value, AffineScalarFunc)]
    names_w_uncert = [key for key, value in kwargs.items() if isinstance(value, AffineScalarFunc)]
    if not pos_w_uncert and (not names_w_uncert):
        return f(*args, **kwargs)
    args_values = list(args)
    for index in pos_w_uncert:
        args_values[index] = args[index].nominal_value
    kwargs_uncert_values = {}
    for name in names_w_uncert:
        value_with_uncert = kwargs[name]
        kwargs_uncert_values[name] = value_with_uncert
        kwargs[name] = value_with_uncert.nominal_value
    f_nominal_value = f(*args_values, **kwargs)
    if not isinstance(f_nominal_value, FLOAT_LIKE_TYPES):
        return NotImplemented
    linear_part = []
    for pos in pos_w_uncert:
        linear_part.append((derivatives_args_index[pos](*args_values, **kwargs), args[pos]._linear_part))
    for name in names_w_uncert:
        derivative = derivatives_all_kwargs.setdefault(name, partial_derivative(f, name))
        linear_part.append((derivative(*args_values, **kwargs), kwargs_uncert_values[name]._linear_part))
    return AffineScalarFunc(f_nominal_value, LinearCombination(linear_part))