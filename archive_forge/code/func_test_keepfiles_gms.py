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
def test_keepfiles_gms(self):
    with SolverFactory('gams', solver_io='gms') as opt:
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x >= 10)
        m.o = Objective(expr=m.x)
        tmpdir = mkdtemp()
        results = opt.solve(m, tmpdir=tmpdir, keepfiles=True)
        self.assertTrue(os.path.exists(tmpdir))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, 'model.gms')))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, 'output.lst')))
        if gdxcc_available:
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_p.gdx')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'results_s.gdx')))
        else:
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'results.dat')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'resultsstat.dat')))
        shutil.rmtree(tmpdir)