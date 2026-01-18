import math
import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.tests.base_test import BaseTest
def assertItemsAlmostEqual(self, a, b, places: int=2) -> None:
    super(TestParamConeProg, self).assertItemsAlmostEqual(a, b, places=places)