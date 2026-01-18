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
def test_fixed_var_sign_py(self):
    with SolverFactory('gams', solver_io='python') as opt:
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.z.fix(-3)
        m.c1 = Constraint(expr=m.x + m.y - m.z == 0)
        m.c2 = Constraint(expr=m.z + m.y - m.z >= -10000)
        m.c3 = Constraint(expr=-3 * m.z + m.y - m.z >= -10000)
        m.c4 = Constraint(expr=-m.z + m.y - m.z >= -10000)
        m.c5 = Constraint(expr=m.x <= 100)
        m.o = Objective(expr=m.x, sense=maximize)
        results = opt.solve(m)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)