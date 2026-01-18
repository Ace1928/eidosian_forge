import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
@unittest.skipIf(sys.platform.lower().startswith('win'), "Cannot (easily) unload a DLL in Windows, so cannot clean up the 'temporary' DLL")
def test_load_local_asl_library(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    LIB = 'test_pyomo_external_gsl.dll'
    model = ConcreteModel()
    model.gamma = ExternalFunction(library=LIB, function='gsl_sf_gamma')
    model.x = Var(initialize=3, bounds=(1e-05, None))
    model.o = Objective(expr=model.gamma(model.x))
    with TempfileManager.new_context() as tempfile:
        dname = tempfile.mkdtemp()
        shutil.copyfile(DLL, os.path.join(dname, LIB))
        with self.assertRaises(OSError):
            value(model.o)
        try:
            orig_dir = os.getcwd()
            os.chdir(dname)
            self.assertAlmostEqual(value(model.o), 2.0, 7)
        finally:
            os.chdir(orig_dir)