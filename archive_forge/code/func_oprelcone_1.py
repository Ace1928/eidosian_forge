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
def oprelcone_1(self, apx_m, apx_k, real) -> STH.SolverTestHelper:
    """
        These tests construct two matrices that commute (imposing all eigenvectors equal)
        and then use the fact that: T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        i.e. T >> Dop(A, B) for an objective that is an increasing function of the
        eigenvalues (which we here take to be the trace), we compute the reference
        objective value as tr(Dop) whose correctness can be seen by writing out
        tr(T)=tr(T-Dop)+tr(Dop), where tr(T-Dop)>=0 because of PSD-ness of (T-Dop),
        and at optimality we have (T-Dop)=0 (the zero matrix of corresponding size)
        For the case that the input matrices commute, Dop takes on a particularly
        simplified form, i.e.: U @ diag(a * log(a/b)) @ U^{-1} (which is implemented
        in the Dop_commute method above)
        """
    temp_obj, temp_con = TestOpRelConeQuad.sum_rel_entr_approx(self.a, self.b, apx_m, apx_k)
    temp_constraints = [con for con in self.base_cons]
    temp_constraints.append(temp_con)
    temp_prob = cp.Problem(temp_obj, temp_constraints)
    temp_prob.solve()
    expect_a = self.a.value
    expect_b = self.b.value
    expect_objective = temp_obj.value
    n = self.n
    if real:
        randmat = self.rng.normal(size=(n, n))
        U = sp.linalg.qr(randmat)[0]
        A = cp.symmetric_wrap(U @ cp.diag(self.a) @ U.T)
        B = cp.symmetric_wrap(U @ cp.diag(self.b) @ U.T)
        T = cp.Variable(shape=(n, n), symmetric=True)
    else:
        randmat = 1j * self.rng.normal(size=(n, n))
        randmat += self.rng.normal(size=(n, n))
        U = sp.linalg.qr(randmat)[0]
        A = cp.hermitian_wrap(U @ cp.diag(self.a) @ U.conj().T)
        B = cp.hermitian_wrap(U @ cp.diag(self.b) @ U.conj().T)
        T = cp.Variable(shape=(n, n), hermitian=True)
    main_con = cp.constraints.OpRelEntrConeQuad(A, B, T, apx_m, apx_k)
    obj = cp.Minimize(trace(T))
    expect_T = TestOpRelConeQuad.Dop_commute(expect_a, expect_b, U)
    con_pairs = [(con, None) for con in self.base_cons]
    con_pairs.append((main_con, None))
    obj_pair = (obj, expect_objective)
    var_pairs = [(T, expect_T)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth