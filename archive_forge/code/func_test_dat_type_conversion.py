from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_dat_type_conversion(self):
    model = AbstractModel()
    model.I = Set()
    model.p = Param(model.I, domain=Any)
    i = model.create_instance(currdir + 'data_types.dat')
    ref = {50: (int, 2), 55: (int, -2), 51: (int, 200), 52: (int, -200), 53: (float, 0.02), 54: (float, -0.02), 10: (float, 1.0), 11: (float, -1.0), 12: (float, 0.1), 13: (float, -0.1), 14: (float, 1.1), 15: (float, -1.1), 20: (float, 200.0), 21: (float, -200.0), 22: (float, 0.02), 23: (float, -0.02), 30: (float, 210.0), 31: (float, -210.0), 32: (float, 0.021), 33: (float, -0.021), 40: (float, 10.0), 41: (float, -10.0), 42: (float, 0.001), 43: (float, -0.001), 1000: (str, 'a_string'), 1001: (str, 'a_string'), 1002: (str, 'a_string'), 1003: (str, 'a " string'), 1004: (str, "a ' string"), 1005: (str, '1234_567'), 1006: (str, '123')}
    for k, v in i.p.items():
        if k in ref:
            err = 'index %s: (%s, %s) does not match ref %s' % (k, type(v), v, ref[k])
            self.assertIs(type(v), ref[k][0], err)
            self.assertEqual(v, ref[k][1], err)
        else:
            n = k // 10
            err = 'index %s: (%s, %s) does not match ref %s' % (k, type(v), v, ref[n])
            self.assertIs(type(v), ref[n][0], err)
            self.assertEqual(v, ref[n][1], err)