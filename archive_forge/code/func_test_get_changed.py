import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_get_changed(self):
    model = ConcreteModel()
    model.t = ContinuousSet(initialize=[1, 2, 3])
    self.assertFalse(model.t.get_changed())
    self.assertEqual(model.t._changed, model.t.get_changed())