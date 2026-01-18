from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
def test_md11a(self):
    cwd = os.getcwd()
    os.chdir(currdir)
    model = AbstractModel()
    model.a = Param()
    model.b = Param()
    model.c = Param()
    model.d = Param()
    instance = model.create_instance(currdir + 'data14.dat', namespaces=['ns1', 'ns2'])
    self.assertEqual(value(instance.a), 1)
    self.assertEqual(value(instance.b), 2)
    self.assertEqual(value(instance.c), 2)
    self.assertEqual(value(instance.d), 2)
    instance = model.create_instance(currdir + 'data14.dat', namespaces=['ns1', 'ns3', 'nsX'])
    self.assertEqual(value(instance.a), 1)
    self.assertEqual(value(instance.b), 100)
    self.assertEqual(value(instance.c), 3)
    self.assertEqual(value(instance.d), 100)
    instance = model.create_instance(currdir + 'data14.dat')
    self.assertEqual(value(instance.a), -1)
    self.assertEqual(value(instance.b), -2)
    self.assertEqual(value(instance.c), -3)
    self.assertEqual(value(instance.d), -4)
    os.chdir(cwd)