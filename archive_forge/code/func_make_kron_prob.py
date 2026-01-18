from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def make_kron_prob(z_dims: Tuple[int], c_dims: Tuple[int], param: bool, var_left: bool, seed: int):
    """
        Construct random nonnegative matrices (C, L) of shapes
        (c_dims, z_dims) respectively. Define an optimization
        problem with a matrix variable of shape z_dims:

            min sum(Z)
            s.t.  kron(Z, C) >= kron(L, C)   ---   if var_left is True
                  kron(C, Z) >= kron(C, L)   ---   if var_left is False
                  Z >= 0

        Regardless of whether var_left is True or False, the optimal
        solution to that problem is Z = L.

        If param is True, then C is defined as a CVXPY Parameter.
        If param is False, then C is a CVXPY Constant.

        A small remark: the constraint that Z >= 0 is redundant.
        It's there because it's easier to set break points that distinguish
        objective canonicalization and constraint canonicalization
        when there's more than one constraint.
        """
    np.random.seed(seed)
    C_value = np.random.rand(*c_dims).round(decimals=2)
    if param:
        C = cp.Parameter(shape=c_dims)
        C.value = C_value
    else:
        C = cp.Constant(C_value)
    Z = cp.Variable(shape=z_dims)
    L = np.random.rand(*Z.shape).round(decimals=2)
    if var_left:
        constraints = [cp.kron(Z, C) >= cp.kron(L, C), Z >= 0]
    else:
        constraints = [cp.kron(C, Z) >= cp.kron(C, L), Z >= 0]
    obj_expr = cp.sum(Z)
    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    return (Z, C, L, prob)