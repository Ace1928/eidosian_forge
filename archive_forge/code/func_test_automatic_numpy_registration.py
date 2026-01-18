import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
@unittest.skipUnless(numpy_available, 'This test requires NumPy')
def test_automatic_numpy_registration(self):
    cmd = 'import pyomo; from pyomo.core.base import Var, Param; from pyomo.core.base.units_container import units; import numpy as np; print(np.float64 in pyomo.common.numeric_types.native_numeric_types); %s; print(np.float64 in pyomo.common.numeric_types.native_numeric_types)'

    def _tester(expr):
        rc = subprocess.run([sys.executable, '-c', cmd % expr], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual((rc.returncode, rc.stdout), (0, 'False\nTrue\n'))
    _tester('Var() <= np.float64(5)')
    _tester('np.float64(5) <= Var()')
    _tester('np.float64(5) + Var()')
    _tester('Var() + np.float64(5)')
    _tester('v = Var(); v.construct(); v.value = np.float64(5)')
    _tester('p = Param(mutable=True); p.construct(); p.value = np.float64(5)')
    _tester('v = Var(units=units.m); v.construct(); v.value = np.float64(5)')
    _tester('p = Param(mutable=True, units=units.m); p.construct(); p.value = np.float64(5)')