import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
@unittest.skipIf(not scipy_available, 'Scipy is not available')
def test_ip2_scipy(self):
    solver = ScipyInterface()
    solver.compute_inertia = True
    self._test_solve_interior_point_2(solver)