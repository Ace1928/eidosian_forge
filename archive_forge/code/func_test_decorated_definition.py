from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_decorated_definition(self):
    model = ConcreteModel()
    model.I = Set(initialize=[1, 2, 3])
    model.x = Var(model.I)

    @model.Constraint()
    def scalar_constraint(m):
        return m.x[1] ** 2 <= 0
    self.assertTrue(hasattr(model, 'scalar_constraint'))
    self.assertIs(model.scalar_constraint.ctype, Constraint)
    self.assertEqual(len(model.scalar_constraint), 1)
    self.assertIs(type(scalar_constraint), types.FunctionType)

    @model.Constraint(model.I)
    def vector_constraint(m, i):
        return m.x[i] ** 2 <= 0
    self.assertTrue(hasattr(model, 'vector_constraint'))
    self.assertIs(model.vector_constraint.ctype, Constraint)
    self.assertEqual(len(model.vector_constraint), 3)
    self.assertIs(type(vector_constraint), types.FunctionType)