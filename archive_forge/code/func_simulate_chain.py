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
def simulate_chain(in_prob, affine, **solve_kwargs):
    reductions = [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()]
    chain = Chain(None, reductions)
    cone_prog, inv_prob2cone = chain.apply(in_prob)
    cone_prog = ConicSolver().format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
    data, inv_data = a2d.Slacks.apply(cone_prog, affine)
    G, h, f, K_dir, K_aff = (data[s.A], data[s.B], data[s.C], data['K_dir'], data['K_aff'])
    G = sp.sparse.csc_matrix(G)
    y = cp.Variable(shape=(G.shape[1],))
    objective = cp.Minimize(f @ y)
    aff_con = TestSlacks.set_affine_constraints(G, h, y, K_aff)
    dir_con = TestSlacks.set_direct_constraints(y, K_dir)
    int_con = TestSlacks.set_integer_constraints(y, data)
    constraints = aff_con + dir_con + int_con
    slack_prob = cp.Problem(objective, constraints)
    slack_prob.solve(**solve_kwargs)
    slack_prims = {a2d.FREE: y[:cone_prog.x.size].value}
    slack_sol = Solution(slack_prob.status, slack_prob.value, slack_prims, None, dict())
    cone_sol = a2d.Slacks.invert(slack_sol, inv_data)
    in_prob_sol = chain.invert(cone_sol, inv_prob2cone)
    in_prob.unpack(in_prob_sol)