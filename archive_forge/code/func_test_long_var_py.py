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
def test_long_var_py(self):
    with SolverFactory('gams', solver_io='python') as opt:
        m = ConcreteModel()
        x = m.a23456789012345678901234567890123456789012345678901234567890123 = Var()
        y = m.b234567890123456789012345678901234567890123456789012345678901234 = Var()
        z = m.c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 = Var()
        w = m.d01234567890 = Var()
        m.c1 = Constraint(expr=x + y + z + w == 0)
        m.c2 = Constraint(expr=x >= 10)
        m.o = Objective(expr=x)
        results = opt.solve(m)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)