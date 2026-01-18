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
@pytest.mark.skipif(not _cylp_checks_isProvenInfeasible(), reason='CyLP <= 0.91.4 has no working integer infeasibility detection')
def test_cbc_mi_lp_5(self) -> None:
    StandardTestLPs.test_mi_lp_5(solver='CBC')