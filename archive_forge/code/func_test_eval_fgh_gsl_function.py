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
def test_eval_fgh_gsl_function(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    model = ConcreteModel()
    model.gamma = ExternalFunction(library=DLL, function='gsl_sf_gamma')
    model.beta = ExternalFunction(library=DLL, function='gsl_sf_beta')
    model.bessel = ExternalFunction(library=DLL, function='gsl_sf_bessel_Jnu')
    f, g, h = model.gamma.evaluate_fgh((2.0,))
    self.assertAlmostEqual(f, 1.0, 7)
    self.assertListsAlmostEqual(g, [0.422784335098467], 7)
    self.assertListsAlmostEqual(h, [0.8236806608528794], 7)
    f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fixed=[1, 1])
    self.assertAlmostEqual(f, 0.11428571428571432, 7)
    self.assertListsAlmostEqual(g, [0.0, 0.0], 7)
    self.assertListsAlmostEqual(h, [0.0, 0.0, 0.0], 7)
    f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fixed=[0, 1])
    self.assertAlmostEqual(f, 0.11428571428571432, 7)
    self.assertListsAlmostEqual(g, [-0.07836734693877555, 0.0], 7)
    self.assertListsAlmostEqual(h, [0.08135276967930034, 0.0, 0.0], 7)
    f, g, h = model.beta.evaluate_fgh((2.5, 2.0))
    self.assertAlmostEqual(f, 0.11428571428571432, 7)
    self.assertListsAlmostEqual(g, [-0.07836734693877555, -0.11040989614412142], 7)
    self.assertListsAlmostEqual(h, [0.08135276967930034, 0.0472839170086535, 0.15194654464270113], 7)
    f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fgh=1)
    self.assertAlmostEqual(f, 0.11428571428571432, 7)
    self.assertListsAlmostEqual(g, [-0.07836734693877555, -0.11040989614412142], 7)
    self.assertIsNone(h)
    f, g, h = model.beta.evaluate_fgh((2.5, 2.0), fgh=0)
    self.assertAlmostEqual(f, 0.11428571428571432, 7)
    self.assertIsNone(g)
    self.assertIsNone(h)
    f, g, h = model.bessel.evaluate_fgh((2.5, 2.0), fixed=[1, 0])
    self.assertAlmostEqual(f, 0.223924531469, 7)
    self.assertListsAlmostEqual(g, [0.0, 0.21138811435101745], 7)
    self.assertListsAlmostEqual(h, [0.0, 0.0, 0.02026349177575621], 7)
    f, g, h = model.gamma.evaluate_fgh((2.0,), fixed=[1])
    self.assertAlmostEqual(f, 1.0, 7)
    self.assertListsAlmostEqual(g, [0.0], 7)
    self.assertListsAlmostEqual(h, [0.0], 7)
    f, g, h = model.bessel.evaluate_fgh((2.5, 2.0), fixed=[1, 1])
    self.assertAlmostEqual(f, 0.223924531469, 7)
    self.assertListsAlmostEqual(g, [0.0, 0.0], 7)
    self.assertListsAlmostEqual(h, [0.0, 0.0, 0.0], 7)