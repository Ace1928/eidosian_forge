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
@pytest.mark.skip(reason='Known bug in ECOS BB')
def test_ecos_bb_mi_socp_1(self) -> None:
    StandardTestSOCPs.test_mi_socp_1(solver='ECOS_BB')