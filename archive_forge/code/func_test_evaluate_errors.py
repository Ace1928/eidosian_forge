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
def test_evaluate_errors(self):
    m = ConcreteModel()
    m.f = ExternalFunction(_f, _g_bad, _h_bad)
    f = m.f.evaluate((5, 7, 11, m.f._fcn_id))
    self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
    with self.assertRaisesRegex(RuntimeError, 'PythonCallbackFunction called with invalid Global ID'):
        f = m.f.evaluate((5, 7, 11, -1))
    with self.assertRaisesRegex(RuntimeError, "External function 'f' returned an invalid derivative vector \\(expected 4, received 5\\)"):
        f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
    with self.assertRaisesRegex(RuntimeError, "External function 'f' returned an invalid Hessian matrix \\(expected 10, received 9\\)"):
        f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=2)