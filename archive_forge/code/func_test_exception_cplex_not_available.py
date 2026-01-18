import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
@unittest.skipIf(pyo.SolverFactory('cplex_persistent').available(exception_flag=False), 'CPLEX available')
def test_exception_cplex_not_available(self):
    self._assert_raises_unavailable_solver('cplex')