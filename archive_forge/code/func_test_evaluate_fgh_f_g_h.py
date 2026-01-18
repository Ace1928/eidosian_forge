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
def test_evaluate_fgh_f_g_h(self):
    m = ConcreteModel()
    m.f = ExternalFunction(_f, _g, _h)
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
    self.assertEqual(h, [2, 3 + 11 ** 2, 0, 2 * 7 * 11, 2 * 5 * 11, 2 * 5 * 7, 0, 0, 0, 0])
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fixed=[0, 1, 0, 1])
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 0, 2 * 5 * 7 * 11, 0])
    self.assertEqual(h, [2, 0, 0, 2 * 7 * 11, 0, 2 * 5 * 7, 0, 0, 0, 0])
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
    self.assertIsNone(h)
    f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    self.assertIsNone(g)
    self.assertIsNone(h)