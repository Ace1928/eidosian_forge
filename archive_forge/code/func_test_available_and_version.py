import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
def test_available_and_version(self):
    solver = pyo.SolverFactory('scipy.root')
    self.assertTrue(solver.available())
    self.assertTrue(solver.license_is_valid())
    sp_version = tuple((int(num) for num in scipy.__version__.split('.')))
    self.assertEqual(sp_version, solver.version())