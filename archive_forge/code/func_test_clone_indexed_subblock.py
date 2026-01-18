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
def test_clone_indexed_subblock(self):
    m = ConcreteModel()

    @m.Block([1, 2, 3])
    def blk(b, i):
        b.IDX = RangeSet(i)
        b.x = Var(b.IDX)
    m.c = Block(rule=m.blk[2].clone())
    self.assertEqual([1, 2], list(m.c.IDX))
    self.assertEqual(list(m.blk[2].IDX), list(m.c.IDX))
    self.assertIsNot(m.blk[2].IDX, m.c.IDX)
    self.assertIsNot(m.blk[2].x, m.c.x)
    self.assertIsNot(m.blk[2].IDX, m.c.x.index_set())
    self.assertIs(m.c.IDX, m.c.x.index_set())
    self.assertIs(m.c.parent_component(), m.c)
    self.assertIs(m.c.parent_block(), m)
    m.c1 = Block()
    m.c1.transfer_attributes_from(m.blk[3].clone())
    self.assertEqual([1, 2, 3], list(m.c1.IDX))
    self.assertEqual(list(m.blk[3].IDX), list(m.c1.IDX))
    self.assertIsNot(m.blk[3].IDX, m.c1.IDX)
    self.assertIsNot(m.blk[3].x, m.c1.x)
    self.assertIsNot(m.blk[3].IDX, m.c1.x.index_set())
    self.assertIs(m.c1.IDX, m.c1.x.index_set())
    self.assertIs(m.c1.parent_component(), m.c1)
    self.assertIs(m.c1.parent_block(), m)

    @m.Block([1, 2, 3])
    def d(b, i):
        return b.model().blk[i].clone()
    for i in [1, 2, 3]:
        self.assertEqual(list(range(1, i + 1)), list(m.d[i].IDX))
        self.assertEqual(list(m.blk[i].IDX), list(m.d[i].IDX))
        self.assertIsNot(m.blk[i].IDX, m.d[i].IDX)
        self.assertIsNot(m.blk[i].x, m.d[i].x)
        self.assertIsNot(m.blk[i].IDX, m.d[i].x.index_set())
        self.assertIs(m.d[i].IDX, m.d[i].x.index_set())
        self.assertIs(m.d[i].parent_component(), m.d)
        self.assertIs(m.d[i].parent_block(), m)