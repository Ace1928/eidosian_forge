import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def test_clone_abstract_model(self):

    def _populate(b, *args):
        b.A = RangeSet(1, 3)
        b.v = Var()
        b.vv = Var(m.A)
        b.p = Param()
    m = AbstractModel()
    _populate(m)
    m.b = Block()
    _populate(m.b)
    m.b.c = Block()
    _populate(m.b.c)
    m.bb = Block(m.A, rule=_populate)
    n = m.clone()
    self.assertNotEqual(id(m), id(n))
    self.assertEqual(id(m), id(m.b.parent_block()))
    self.assertEqual(id(m), id(m.bb.parent_block()))
    self.assertEqual(id(n), id(n.b.parent_block()))
    self.assertEqual(id(n), id(n.bb.parent_block()))
    for x, y in ((m, n), (m.b, n.b), (m.b.c, n.b.c)):
        self.assertNotEqual(id(x), id(y))
        self.assertNotEqual(id(x.parent_block()), id(x))
        self.assertNotEqual(id(y.parent_block()), id(y))
        self.assertEqual(id(x.A.parent_block()), id(x))
        self.assertEqual(id(x.v.parent_block()), id(x))
        self.assertEqual(id(x.vv.parent_block()), id(x))
        self.assertEqual(id(x.p.parent_block()), id(x))
        self.assertEqual(id(y.A.parent_block()), id(y))
        self.assertEqual(id(y.v.parent_block()), id(y))
        self.assertEqual(id(y.vv.parent_block()), id(y))
        self.assertEqual(id(y.p.parent_block()), id(y))