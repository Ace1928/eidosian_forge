import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
@unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
def test_custom_solver_name(self):
    self.instance = base.SolverBase(name='my_unique_name')
    self.assertEqual(self.instance.name, 'my_unique_name')