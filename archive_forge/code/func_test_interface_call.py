import pyomo.common.unittest as unittest
from pyomo.opt import TerminationCondition, SolutionStatus, check_available_solvers
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_interface_call(self):
    interface_instance = type(pyo.SolverFactory('mosek_direct'))
    alt_1 = pyo.SolverFactory('mosek')
    alt_2 = pyo.SolverFactory('mosek', solver_io='python')
    alt_3 = pyo.SolverFactory('mosek', solver_io='direct')
    self.assertIsInstance(alt_1, interface_instance)
    self.assertIsInstance(alt_2, interface_instance)
    self.assertIsInstance(alt_3, interface_instance)