import pickle
import math
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint, IntegerSet
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable, variable_tuple
from pyomo.core.kernel.block import block
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.conic import (
def test_build_linking_constraints(self):
    c = _build_linking_constraints([], [])
    self.assertIs(type(c), constraint_tuple)
    self.assertEqual(len(c), 0)
    c = _build_linking_constraints([None], [variable()])
    self.assertIs(type(c), constraint_tuple)
    self.assertEqual(len(c), 0)
    v = [1, data_expression(), variable(), expression(expr=1.0)]
    vaux = [variable(), variable(), variable(), variable()]
    c = _build_linking_constraints(v, vaux)
    self.assertIs(type(c), constraint_tuple)
    self.assertEqual(len(c), 4)
    self.assertIs(type(c[0]), linear_constraint)
    self.assertEqual(c[0].rhs, 1)
    self.assertEqual(len(list(c[0].terms)), 1)
    self.assertIs(list(c[0].terms)[0][0], vaux[0])
    self.assertEqual(list(c[0].terms)[0][1], 1)
    self.assertIs(type(c[1]), linear_constraint)
    self.assertIs(c[1].rhs, v[1])
    self.assertEqual(len(list(c[1].terms)), 1)
    self.assertIs(list(c[1].terms)[0][0], vaux[1])
    self.assertEqual(list(c[1].terms)[0][1], 1)
    self.assertIs(type(c[2]), linear_constraint)
    self.assertEqual(c[2].rhs, 0)
    self.assertEqual(len(list(c[2].terms)), 2)
    self.assertIs(list(c[2].terms)[0][0], vaux[2])
    self.assertEqual(list(c[2].terms)[0][1], 1)
    self.assertIs(list(c[2].terms)[1][0], v[2])
    self.assertEqual(list(c[2].terms)[1][1], -1)
    self.assertIs(type(c[3]), constraint)
    self.assertEqual(c[3].rhs, 0)
    from pyomo.repn import generate_standard_repn
    repn = generate_standard_repn(c[3].body)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], vaux[3])
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertEqual(repn.constant, -1)