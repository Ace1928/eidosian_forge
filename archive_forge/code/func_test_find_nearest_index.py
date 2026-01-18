import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_find_nearest_index(self):
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 5))
    i = m.time.find_nearest_index(1)
    self.assertEqual(i, 1)
    i = m.time.find_nearest_index(1, tolerance=0.5)
    self.assertEqual(i, None)
    i = m.time.find_nearest_index(-0.01, tolerance=0.1)
    self.assertEqual(i, 1)
    i = m.time.find_nearest_index(-0.01, tolerance=0.001)
    self.assertEqual(i, None)
    i = m.time.find_nearest_index(6, tolerance=2)
    self.assertEqual(i, 2)
    i = m.time.find_nearest_index(6, tolerance=1)
    self.assertEqual(i, 2)
    i = m.time.find_nearest_index(2.5)
    self.assertEqual(i, 1)
    m.del_component(m.time)
    init_list = []
    for i in range(5):
        i0 = float(i)
        i1 = round((i + 0.15) * 10000.0) / 10000.0
        i2 = round((i + 0.64) * 10000.0) / 10000.0
        init_list.extend([i, i1, i2])
    init_list.append(5.0)
    m.time = ContinuousSet(initialize=init_list)
    i = m.time.find_nearest_index(1.01, tolerance=0.1)
    self.assertEqual(i, 4)
    i = m.time.find_nearest_index(1.01, tolerance=0.001)
    self.assertEqual(i, None)
    i = m.time.find_nearest_index(3.5)
    self.assertEqual(i, 12)
    i = m.time.find_nearest_index(3.5, tolerance=0.1)
    self.assertEqual(i, None)
    i = m.time.find_nearest_index(-1)
    self.assertEqual(i, 1)
    i = m.time.find_nearest_index(-1, tolerance=1)
    self.assertEqual(i, 1)
    i = m.time.find_nearest_index(5.5)
    self.assertEqual(i, 16)
    i = m.time.find_nearest_index(5.5, tolerance=0.49)
    self.assertEqual(i, None)
    i = m.time.find_nearest_index(2.64, tolerance=1e-08)
    self.assertEqual(i, 9)
    i = m.time.find_nearest_index(2.64, tolerance=0)
    self.assertEqual(i, 9)
    i = m.time.find_nearest_index(5, tolerance=0)
    self.assertEqual(i, 16)
    i = m.time.find_nearest_index(0, tolerance=0)
    self.assertEqual(i, 1)