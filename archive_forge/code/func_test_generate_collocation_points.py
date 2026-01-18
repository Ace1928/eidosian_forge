import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_generate_collocation_points(self):
    m = ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1])
    m.t2 = ContinuousSet(initialize=[0, 2, 4, 6])
    tau1 = [1]
    oldt = sorted(m.t)
    generate_colloc_points(m.t, tau1)
    self.assertTrue(oldt == sorted(m.t))
    self.assertFalse(m.t.get_changed())
    tau1 = [0.5]
    oldt = sorted(m.t)
    generate_colloc_points(m.t, tau1)
    self.assertFalse(oldt == sorted(m.t))
    self.assertTrue(m.t.get_changed())
    self.assertTrue([0, 0.5, 1] == sorted(m.t))
    tau2 = [0.2, 0.3, 0.7, 0.8, 1]
    generate_colloc_points(m.t, tau2)
    self.assertTrue(len(m.t) == 11)
    self.assertTrue([0, 0.1, 0.15, 0.35, 0.4, 0.5, 0.6, 0.65, 0.85, 0.9, 1] == sorted(m.t))
    generate_colloc_points(m.t2, tau2)
    self.assertTrue(len(m.t2) == 16)
    self.assertTrue(m.t2.get_changed())
    t = sorted(m.t2)
    self.assertTrue(t[1] == 0.4)
    self.assertTrue(t[13] == 5.4)