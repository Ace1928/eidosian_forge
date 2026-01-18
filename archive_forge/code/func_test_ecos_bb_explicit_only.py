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
def test_ecos_bb_explicit_only(self) -> None:
    """Test that ECOS_BB isn't chosen by default.
        """
    x = cp.Variable(1, name='x', integer=True)
    objective = cp.Minimize(cp.sum(x))
    prob = cp.Problem(objective, [x >= 0])
    if INSTALLED_MI_SOLVERS != [cp.ECOS_BB]:
        prob.solve()
        assert prob.solver_stats.solver_name != cp.ECOS_BB
    else:
        with pytest.raises(cp.error.SolverError, match='You need a mixed-integer solver for this model'):
            prob.solve()