import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
def test_check_expr_eval_py(self):
    with SolverFactory('gams', solver_io='python') as opt:
        m = ConcreteModel()
        m.x = Var()
        m.e = Expression(expr=log10(m.x) + 5)
        m.c = Constraint(expr=m.x >= 10)
        m.o = Objective(expr=m.e)
        self.assertRaises(GamsExceptionExecution, opt.solve, m)