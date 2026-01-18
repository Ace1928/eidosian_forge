import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
def test_logfile_relative(self):
    cwd = os.getcwd()
    with TempfileManager:
        tmpdir = TempfileManager.create_tempdir()
        os.chdir(tmpdir)
        try:
            self.logfile = 'test-gams.log'
            with SolverFactory('gams', solver_io='python') as opt:
                with capture_output() as output:
                    opt.solve(self.m, logfile=self.logfile)
            self._check_stdout(output.getvalue(), exists=False)
            self._check_logfile(exists=True)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, self.logfile)))
        finally:
            os.chdir(cwd)