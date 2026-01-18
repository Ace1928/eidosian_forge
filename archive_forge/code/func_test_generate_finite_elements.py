import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_generate_finite_elements(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.t2 = ContinuousSet(bounds=(0, 10))
    m.t3 = ContinuousSet(bounds=(0, 1))
    oldt = sorted(m.t)
    generate_finite_elements(m.t, 1)
    self.assertTrue(oldt == sorted(m.t))
    self.assertFalse(m.t.get_changed())
    generate_finite_elements(m.t, 2)
    self.assertFalse(oldt == sorted(m.t))
    self.assertTrue(m.t.get_changed())
    self.assertTrue([0, 5.0, 10] == sorted(m.t))
    generate_finite_elements(m.t, 3)
    self.assertTrue([0, 2.5, 5.0, 10] == sorted(m.t))
    generate_finite_elements(m.t, 5)
    self.assertTrue([0, 1.25, 2.5, 5.0, 7.5, 10] == sorted(m.t))
    generate_finite_elements(m.t2, 10)
    self.assertTrue(len(m.t2) == 11)
    self.assertTrue([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] == sorted(m.t2))
    generate_finite_elements(m.t3, 7)
    self.assertTrue(len(m.t3) == 8)
    t = sorted(m.t3)
    print(t[1])
    self.assertTrue(t[1] == 0.142857)