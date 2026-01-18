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
def test_eval_gsl_function(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    model = ConcreteModel()
    model.gamma = ExternalFunction(library=DLL, function='gsl_sf_gamma')
    model.bessel = ExternalFunction(library=DLL, function='gsl_sf_bessel_Jnu')
    model.x = Var(initialize=3, bounds=(1e-05, None))
    model.o = Objective(expr=model.gamma(model.x))
    self.assertAlmostEqual(value(model.o), 2.0, 7)
    f = model.bessel.evaluate((0.5, 2.0))
    self.assertAlmostEqual(f, 0.5130161365618272, 7)