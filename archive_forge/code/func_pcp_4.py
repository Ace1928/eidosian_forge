import unittest
import numpy as np
import pytest
import scipy as sp
import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
@staticmethod
def pcp_4(ceei: bool=True):
    """
        A power cone formulation of a Fisher market equilibrium pricing model.
        ceei = Competitive Equilibrium from Equal Incomes
        """
    np.random.seed(0)
    n_buyer = 4
    n_items = 6
    V = np.random.rand(n_buyer, n_items)
    X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
    u = cp.sum(cp.multiply(V, X), axis=1)
    if ceei:
        b = np.ones(n_buyer) / n_buyer
    else:
        b = np.array([0.3, 0.15, 0.2, 0.35])
    log_objective = cp.Maximize(cp.sum(cp.multiply(b, cp.log(u))))
    log_cons = [cp.sum(X, axis=0) <= 1]
    log_prob = cp.Problem(log_objective, log_cons)
    log_prob.solve(solver='SCS', eps=1e-08)
    expect_X = X.value
    z = cp.Variable()
    pow_objective = (cp.Maximize(z), np.exp(log_prob.value))
    pow_cons = [(cp.sum(X, axis=0) <= 1, None), (PowConeND(W=u, z=z, alpha=b), None)]
    pow_vars = [(X, expect_X)]
    sth = STH.SolverTestHelper(pow_objective, pow_vars, pow_cons)
    return sth