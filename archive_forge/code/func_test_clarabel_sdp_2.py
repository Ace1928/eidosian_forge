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
def test_clarabel_sdp_2(self) -> None:
    places = 3
    sth = sths.sdp_2()
    sth.solve('CLARABEL')
    sth.verify_objective(places)
    sth.check_primal_feasibility(places)
    sth.check_complementarity(places)
    sth.check_dual_domains(places)