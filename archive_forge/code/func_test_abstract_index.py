import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_abstract_index(self):
    model = AbstractModel()
    model.A = Set()
    model.B = Set()
    model.C = model.A | model.B
    model.x = Objective(model.C)