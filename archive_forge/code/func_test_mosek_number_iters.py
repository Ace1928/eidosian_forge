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
def test_mosek_number_iters(self) -> None:
    sth = sths.lp_5()
    sth.solve(solver=cp.MOSEK)
    assert sth.prob.solver_stats.num_iters >= 0
    assert sth.prob.solver_stats.extra_stats['mio_intpnt_iter'] == 0
    assert sth.prob.solver_stats.extra_stats['mio_simplex_iter'] == 0