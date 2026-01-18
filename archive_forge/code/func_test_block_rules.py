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
def test_block_rules(self):
    m = ConcreteModel()
    m.I = Set()
    _rule_ = []

    def _block_rule(b, i):
        _rule_.append(i)
        b.x = Var(range(i))
    m.b = Block(m.I, rule=_block_rule)
    self.assertEqual(_rule_, [])
    m.I.update([1, 3, 5])
    _b = m.b[3]
    self.assertEqual(len(m.b), 1)
    self.assertEqual(_rule_, [3])
    self.assertIn('x', _b.component_map())
    self.assertIn('x', m.b[3].component_map())
    _tmp = Block()
    _tmp.y = Var(range(3))
    m.b[5].transfer_attributes_from(_tmp)
    self.assertEqual(len(m.b), 2)
    self.assertEqual(_rule_, [3, 5])
    self.assertIn('x', m.b[5].component_map())
    self.assertIn('y', m.b[5].component_map())
    _tmp = Block()
    _tmp.y = Var(range(3))
    with self.assertRaisesRegex(RuntimeError, 'Block components do not support assignment or set_value'):
        m.b[1] = _tmp
    self.assertEqual(len(m.b), 2)
    self.assertEqual(_rule_, [3, 5])

    def _bb_rule(b, i, j):
        _rule_.append((i, j))
        b.x = Var(RangeSet(i))
        b.y = Var(RangeSet(j))
    m.bb = Block(m.I, NonNegativeIntegers, rule=_bb_rule)
    self.assertEqual(_rule_, [3, 5])
    _b = m.bb[3, 5]
    self.assertEqual(_rule_, [3, 5, (3, 5)])
    self.assertEqual(len(m.bb), 1)
    self.assertEqual(len(_b.x), 3)
    self.assertEqual(len(_b.y), 5)