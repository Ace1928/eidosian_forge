import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def get_simple_problem(self):
    """Example problem that can be used within additional tests."""
    x = cp.Variable()
    y = cp.Variable()
    constraints = [x >= 0, y >= 1, x + y <= 4]
    obj = cp.Maximize(x)
    prob = cp.Problem(obj, constraints)
    return prob