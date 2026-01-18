from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor
def test_issue_2402_scalar_parameter():
    """
    This is the problem reported in #2402, failing to solve when two parameters
    are used on quadratic forms with the same variable.
    """
    r = np.array([-0.48, 0.11, 0.09, -0.39, 0.03])
    Sigma = np.array([[0.00024, 0.00013, 0.0002, 0.00016, 0.0002], [0.00013, 0.00028, 0.00021, 0.00017, 0.00015], [0.0002, 0.00021, 0.00058, 0.00033, 0.00023], [0.00016, 0.00017, 0.00033, 0.00069, 0.00021], [0.0002, 0.00015, 0.00023, 0.00021, 0.00036]])
    w = cp.Variable(5)
    risk_aversion = cp.Parameter(value=1.0, nonneg=True)
    ridge_coef = cp.Parameter(value=0.0, nonneg=True)
    obj_func = r @ w - risk_aversion * cp.quad_form(w, Sigma) - ridge_coef * cp.sum_squares(w)
    objective = cp.Maximize(obj_func)
    fixed_w = np.array([10, 11, 12, 13, 14])
    constraints = [w == fixed_w]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    expected_value = r @ fixed_w - risk_aversion.value * np.dot(fixed_w, np.dot(Sigma, fixed_w)) - ridge_coef.value * np.sum(np.square(fixed_w))
    assert np.isclose(prob.value, expected_value)
    assert np.allclose(w.value, fixed_w)