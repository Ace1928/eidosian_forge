import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
def test_dat_parser(self):
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=list(range(5)))
    m.a_long_var_name = pyo.Var(m.S, bounds=(0, 1), initialize=1)
    m.obj = pyo.Objective(expr=2000 * pyo.summation(m.a_long_var_name), sense=pyo.maximize)
    solver = pyo.SolverFactory('gams:conopt')
    res = solver.solve(m, symbolic_solver_labels=True, load_solutions=False, io_options={'put_results_format': 'dat'})
    self.assertEqual(res.solution[0].Objective['obj']['Value'], 10000)
    for i in range(5):
        self.assertEqual(res.solution[0].Variable[f'a_long_var_name_{i}_']['Value'], 1)