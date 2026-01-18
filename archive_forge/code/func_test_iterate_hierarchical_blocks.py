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
def test_iterate_hierarchical_blocks(self):

    def def_var(b, *args):
        b.x = Var()

    def init_block(b):
        b.c = Block([1, 2], rule=def_var)
        b.e = Disjunct([1, 2], rule=def_var)
        b.b = Block(rule=def_var)
        b.d = Disjunct(rule=def_var)
    m = ConcreteModel()
    m.x = Var()
    init_block(m)
    init_block(m.b)
    init_block(m.c[1])
    init_block(m.c[2])
    init_block(m.d)
    init_block(m.e[1])
    init_block(m.e[2])
    ref = [x.name for x in (m, m.c[1], m.c[1].c[1], m.c[1].c[2], m.c[1].b, m.c[2], m.c[2].c[1], m.c[2].c[2], m.c[2].b, m.b, m.b.c[1], m.b.c[2], m.b.b)]
    test = list((x.name for x in m.block_data_objects()))
    self.assertEqual(test, ref)
    test = list((x.name for x in m.block_data_objects(descend_into=Block)))
    self.assertEqual(test, ref)
    test = list((x.name for x in m.block_data_objects(descend_into=(Block,))))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m, m.e[1], m.e[1].e[1], m.e[1].e[2], m.e[1].d, m.e[2], m.e[2].e[1], m.e[2].e[2], m.e[2].d, m.d, m.d.e[1], m.d.e[2], m.d.d)]
    test = list((x.name for x in m.block_data_objects(descend_into=(Disjunct,))))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m.d, m.d.e[1], m.d.e[2], m.d.d)]
    test = list((x.name for x in m.d.block_data_objects(descend_into=(Disjunct,))))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m, m.c[1], m.c[1].c[1], m.c[1].c[2], m.c[1].e[1], m.c[1].e[2], m.c[1].b, m.c[1].d, m.c[2], m.c[2].c[1], m.c[2].c[2], m.c[2].e[1], m.c[2].e[2], m.c[2].b, m.c[2].d, m.e[1], m.e[1].c[1], m.e[1].c[2], m.e[1].e[1], m.e[1].e[2], m.e[1].b, m.e[1].d, m.e[2], m.e[2].c[1], m.e[2].c[2], m.e[2].e[1], m.e[2].e[2], m.e[2].b, m.e[2].d, m.b, m.b.c[1], m.b.c[2], m.b.e[1], m.b.e[2], m.b.b, m.b.d, m.d, m.d.c[1], m.d.c[2], m.d.e[1], m.d.e[2], m.d.b, m.d.d)]
    test = list((x.name for x in m.block_data_objects(descend_into=(Block, Disjunct))))
    self.assertEqual(test, ref)
    test = list((x.name for x in m.block_data_objects(descend_into=(Disjunct, Block))))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m.x, m.c[1].x, m.c[1].c[1].x, m.c[1].c[2].x, m.c[1].b.x, m.c[2].x, m.c[2].c[1].x, m.c[2].c[2].x, m.c[2].b.x, m.b.x, m.b.c[1].x, m.b.c[2].x, m.b.b.x)]
    test = list((x.name for x in m.component_data_objects(Var)))
    self.assertEqual(test, ref)
    test = list((x.name for x in m.component_data_objects(Var, descend_into=Block)))
    self.assertEqual(test, ref)
    test = list((x.name for x in m.component_data_objects(Var, descend_into=(Block,))))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m.x, m.e[1].binary_indicator_var, m.e[1].x, m.e[1].e[1].binary_indicator_var, m.e[1].e[1].x, m.e[1].e[2].binary_indicator_var, m.e[1].e[2].x, m.e[1].d.binary_indicator_var, m.e[1].d.x, m.e[2].binary_indicator_var, m.e[2].x, m.e[2].e[1].binary_indicator_var, m.e[2].e[1].x, m.e[2].e[2].binary_indicator_var, m.e[2].e[2].x, m.e[2].d.binary_indicator_var, m.e[2].d.x, m.d.binary_indicator_var, m.d.x, m.d.e[1].binary_indicator_var, m.d.e[1].x, m.d.e[2].binary_indicator_var, m.d.e[2].x, m.d.d.binary_indicator_var, m.d.d.x)]
    test = list((x.name for x in m.component_data_objects(Var, descend_into=Disjunct)))
    self.assertEqual(test, ref)
    ref = [x.name for x in (m.x, m.c[1].x, m.c[1].c[1].x, m.c[1].c[2].x, m.c[1].e[1].binary_indicator_var, m.c[1].e[1].x, m.c[1].e[2].binary_indicator_var, m.c[1].e[2].x, m.c[1].b.x, m.c[1].d.binary_indicator_var, m.c[1].d.x, m.c[2].x, m.c[2].c[1].x, m.c[2].c[2].x, m.c[2].e[1].binary_indicator_var, m.c[2].e[1].x, m.c[2].e[2].binary_indicator_var, m.c[2].e[2].x, m.c[2].b.x, m.c[2].d.binary_indicator_var, m.c[2].d.x, m.e[1].binary_indicator_var, m.e[1].x, m.e[1].c[1].x, m.e[1].c[2].x, m.e[1].e[1].binary_indicator_var, m.e[1].e[1].x, m.e[1].e[2].binary_indicator_var, m.e[1].e[2].x, m.e[1].b.x, m.e[1].d.binary_indicator_var, m.e[1].d.x, m.e[2].binary_indicator_var, m.e[2].x, m.e[2].c[1].x, m.e[2].c[2].x, m.e[2].e[1].binary_indicator_var, m.e[2].e[1].x, m.e[2].e[2].binary_indicator_var, m.e[2].e[2].x, m.e[2].b.x, m.e[2].d.binary_indicator_var, m.e[2].d.x, m.b.x, m.b.c[1].x, m.b.c[2].x, m.b.e[1].binary_indicator_var, m.b.e[1].x, m.b.e[2].binary_indicator_var, m.b.e[2].x, m.b.b.x, m.b.d.binary_indicator_var, m.b.d.x, m.d.binary_indicator_var, m.d.x, m.d.c[1].x, m.d.c[2].x, m.d.e[1].binary_indicator_var, m.d.e[1].x, m.d.e[2].binary_indicator_var, m.d.e[2].x, m.d.b.x, m.d.d.binary_indicator_var, m.d.d.x)]
    test = list((x.name for x in m.component_data_objects(Var, descend_into=(Block, Disjunct))))
    self.assertEqual(test, ref)