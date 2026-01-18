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
def test_gurobi_lp_3(self) -> None:
    sth = sths.lp_3()
    with self.assertWarns(Warning):
        sth.prob.solve(solver='GUROBI')
        self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)
    StandardTestLPs.test_lp_3(solver='GUROBI', InfUnbdInfo=1)
    StandardTestLPs.test_lp_3(solver='GUROBI', reoptimize=True)