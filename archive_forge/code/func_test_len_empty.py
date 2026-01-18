import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_len_empty(self):
    """Test len method"""
    model = ConcreteModel()
    model.o = Objective()
    self.assertEqual(len(model.o), 0)