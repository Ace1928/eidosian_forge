import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_expand_connector(self):
    model = AbstractModel()
    model.A = Set()

    def _b_rule(b, id):
        b.X = Var()
        b.PORT = Connector()
        b.PORT.add(b.X)
    model.B = Block(model.A, rule=_b_rule)

    def _c_rule(m, a):
        return m.B[a].PORT == m.B[(a + 1) % 2].PORT
    model.C = Constraint(model.A, rule=_c_rule)
    instance = model.create_instance({None: {'A': {None: [0, 1]}}})