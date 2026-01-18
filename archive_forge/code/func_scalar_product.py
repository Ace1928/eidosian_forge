from __future__ import division
import operator as op
from functools import reduce
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import (
from cvxpy.expressions.expression import Expression
def scalar_product(x, y):
    """
    Return the standard inner product (or "scalar product") of (x,y).

    Parameters
    ----------
    x : Expression, int, float, NumPy ndarray, or nested list thereof.
        The conjugate-linear argument to the inner product.
    y : Expression, int, float, NumPy ndarray, or nested list thereof.
        The linear argument to the inner product.

    Returns
    -------
    expr : Expression
        The standard inner product of (x,y), conjugate-linear in x.
        We always have ``expr.shape == ()``.

    Notes
    -----
    The arguments ``x`` and ``y`` can be nested lists; these lists
    will be flattened independently of one another.

    For example, if ``x = [[a],[b]]`` and  ``y = [c, d]`` (with ``a,b,c,d``
    real scalars), then this function returns an Expression representing
    ``a * c + b * d``.
    """
    x = deep_flatten(x)
    y = deep_flatten(y)
    prod = multiply(conj(x), y)
    return cvxpy_sum(prod)