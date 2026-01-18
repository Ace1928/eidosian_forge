import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.initialization import (
def test_get_inconsistent_initial_conditions(self):
    m = make_model()
    inconsistent = get_inconsistent_initial_conditions(m, m.time)
    self.assertIn(m.fs.b1.con[m.time[1], m.space[1]], inconsistent)
    self.assertIn(m.fs.b2[m.time[1], m.space[1]].b3['a'].con['d'], inconsistent)
    self.assertIn(m.fs.con1[m.time[1]], inconsistent)
    self.assertNotIn(m.fs.con2[m.space[1]], inconsistent)