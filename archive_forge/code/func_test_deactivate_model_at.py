import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.set_utils import (
def test_deactivate_model_at(self):
    m = make_model()
    deactivate_model_at(m, m.time, m.time[2])
    self.assertTrue(m.fs.con1[m.time[1]].active)
    self.assertFalse(m.fs.con1[m.time[2]].active)
    self.assertTrue(m.fs.con2[m.space[1]].active)
    self.assertFalse(m.fs.b1.con[m.time[2], m.space[1]].active)
    self.assertFalse(m.fs.b2[m.time[2], m.space.last()].active)
    self.assertTrue(m.fs.b2[m.time[2], m.space.last()].b3['a'].con['e'].active)
    deactivate_model_at(m, m.time, [m.time[1], m.time[3]])
    self.assertFalse(m.fs.con1[m.time[1]].active)
    self.assertFalse(m.fs.con1[m.time[3]].active)
    self.assertFalse(m.fs.b1.con[m.time[1], m.space[1]].active)
    self.assertFalse(m.fs.b1.con[m.time[3], m.space[1]].active)
    with self.assertRaises(KeyError):
        deactivate_model_at(m, m.time, m.time[1], allow_skip=False, suppress_warnings=True)