import collections
import itertools
import math
import numpy as np
import numpy.linalg as LA
import pytest
import cvxpy as cp
import cvxpy.interface as intf
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.settings import CVXOPT, ECOS, MOSEK, OSQP, ROBUST_KKTSOLVER, SCS
def log_sum_exp_axis_1(x):
    return cp.log_sum_exp(x, axis=1)