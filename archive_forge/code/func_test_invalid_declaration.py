import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_invalid_declaration(self):
    model = ConcreteModel()
    model.s = Set(initialize=[1, 2, 3])
    with self.assertRaises(TypeError):
        model.t = ContinuousSet(model.s, bounds=(0, 1))
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(bounds=(0, 0))
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(initialize=[1])
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(bounds=(None, 1))
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(bounds=(0, None))
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(initialize=[(1, 2), (3, 4)])
    model = ConcreteModel()
    with self.assertRaises(ValueError):
        model.t = ContinuousSet(initialize=['foo', 'bar'])