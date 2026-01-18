import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_duplicate_construct(self):
    m = ConcreteModel()
    m.t = ContinuousSet(initialize=[1, 2, 3])
    self.assertEqual(m.t, [1, 2, 3])
    self.assertEqual(m.t._fe, [1, 2, 3])
    m.t.add(1.5)
    m.t.add(2.5)
    self.assertEqual(m.t, [1, 1.5, 2, 2.5, 3])
    self.assertEqual(m.t._fe, [1, 2, 3])
    m.t.construct()
    self.assertEqual(m.t, [1, 1.5, 2, 2.5, 3])
    self.assertEqual(m.t._fe, [1, 2, 3])