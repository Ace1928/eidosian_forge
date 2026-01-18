import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
def test_default_instantiation(self):
    opt = ipopt.Ipopt()
    self.assertFalse(opt.is_persistent())
    self.assertIsNotNone(opt.version())
    self.assertEqual(opt.name, 'ipopt')
    self.assertEqual(opt.CONFIG, opt.config)
    self.assertTrue(opt.available())