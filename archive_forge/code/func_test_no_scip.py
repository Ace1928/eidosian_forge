import subprocess
import sys
from os.path import join, exists, splitext
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
import pyomo.solvers.plugins.solvers.SCIPAMPL
def test_no_scip(self):
    self.set_solvers(scip=None)
    scip = SolverFactory('scip', solver_io='nl')
    scip_executable = scip.executable()
    self.assertIs(scip_executable, self.executable_paths['scipampl'])
    self.assertEqual(1, self.path.call_count)
    self.assertEqual(0, self.run.call_count)
    self.available.assert_called()
    self.assertEqual((7, 0, 3, 0), scip._get_version())
    self.assertEqual(1, self.run.call_count)
    scip.executable()
    self.assertEqual(1, self.run.call_count)
    self.assertTrue(scip.available())