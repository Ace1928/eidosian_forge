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
def check_solver(prob, solver_name) -> bool:
    """Can the solver solve the problem?
    """
    atom_str = str(prob.objective.args[0])
    for bad_atom_name in KNOWN_SOLVER_ERRORS[solver_name]:
        if bad_atom_name in atom_str:
            return False
    try:
        if solver_name == ROBUST_CVXOPT:
            solver_name = CVXOPT
        prob._construct_chain(solver=solver_name)
        return True
    except SolverError:
        return False
    except Exception:
        raise