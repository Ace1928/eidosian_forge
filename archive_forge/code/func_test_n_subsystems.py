import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
def test_n_subsystems(self):
    SolverClass = self.get_solver_class()
    fcn = ImplicitFunction1()
    variables = fcn.get_variables()
    parameters = fcn.get_parameters()
    equations = fcn.get_equations()
    solver = SolverClass(variables, equations, parameters)
    self.assertEqual(solver.n_subsystems(), 2)