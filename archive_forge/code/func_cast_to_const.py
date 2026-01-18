import abc
import warnings
from functools import wraps
from typing import Tuple
import numpy as np
import cvxpy.settings as s
import cvxpy.utilities as u
import cvxpy.utilities.key_utils as ku
import cvxpy.utilities.performance_utils as perf
from cvxpy import error
from cvxpy.constraints import PSD, Equality, Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
from cvxpy.utilities.shape import size_from_shape
@staticmethod
def cast_to_const(expr: 'Expression'):
    """Converts a non-Expression to a Constant.
        """
    if isinstance(expr, list):
        for elem in expr:
            if isinstance(elem, Expression):
                raise ValueError('The input must be a single CVXPY Expression, not a list. Combine Expressions using atoms such as bmat, hstack, and vstack.')
    return expr if isinstance(expr, Expression) else cvxtypes.constant()(expr)