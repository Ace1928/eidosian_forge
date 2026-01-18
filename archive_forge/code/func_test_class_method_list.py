import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
def test_class_method_list(self):
    expected_list = ['available', 'license_is_valid', 'solve']
    method_list = [method for method in dir(base.LegacySolverWrapper) if method.startswith('_') is False]
    self.assertEqual(sorted(expected_list), sorted(method_list))