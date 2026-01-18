import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def diffcp_sdp():
    X = cp.Variable((n, n), PSD=True)
    objective = cp.trace(cp.matmul(C, X))
    constraints = [cp.trace(cp.matmul(As[i], X)) == Bs[i] for i in range(p)]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.get_problem_data(cp.SCS)