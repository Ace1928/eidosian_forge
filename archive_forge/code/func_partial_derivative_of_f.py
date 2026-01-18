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
def partial_derivative_of_f(*args, **kwargs):
    """
        Partial derivative, calculated with the (-epsilon, +epsilon)
        method, which is more precise than the (0, +epsilon) method.
        """
    if change_kwargs:
        args_with_var = kwargs
    else:
        args_with_var = list(args)
    step = STEP_SIZE * abs(args_with_var[arg_ref])
    if not step:
        step = STEP_SIZE
    args_with_var[arg_ref] += step
    if change_kwargs:
        shifted_f_plus = f(*args, **args_with_var)
    else:
        shifted_f_plus = f(*args_with_var, **kwargs)
    args_with_var[arg_ref] -= 2 * step
    if change_kwargs:
        shifted_f_minus = f(*args, **args_with_var)
    else:
        shifted_f_minus = f(*args_with_var, **kwargs)
    return (shifted_f_plus - shifted_f_minus) / 2 / step