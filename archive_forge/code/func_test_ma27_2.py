import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.interior_point.interior_point import (
from pyomo.contrib.interior_point.interface import InteriorPointInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
@unittest.skipIf(not ma27_available, 'MA27 is not available')
def test_ma27_2(self):
    solver = InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
    self._test_regularization_2(solver)