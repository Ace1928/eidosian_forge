import tempfile
import os
import pickle
import random
import collections
import itertools
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import (
from pyomo.common.collections import ComponentMap
from pyomo.core.kernel.suffix import suffix
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.parameter import parameter, parameter_dict, parameter_list
from pyomo.core.kernel.expression import (
from pyomo.core.kernel.objective import objective, objective_dict, objective_list
from pyomo.core.kernel.variable import IVariable, variable, variable_dict, variable_list
from pyomo.core.kernel.block import IBlock, block, block_dict, block_tuple, block_list
from pyomo.core.kernel.sos import sos
from pyomo.opt.results import Solution
def test_clone2(self):
    b = block()
    b.v = variable()
    b.vdict = variable_dict(((i, variable()) for i in range(10)))
    b.vlist = variable_list((variable() for i in range(10)))
    b.o = objective(b.v + b.vdict[0] + b.vlist[0])
    b.odict = objective_dict(((i, objective(b.v + b.vdict[i])) for i in b.vdict))
    b.olist = objective_list((objective(b.v + v_) for i, v_ in enumerate(b.vdict)))
    b.c = constraint(b.v >= 1)
    b.cdict = constraint_dict(((i, constraint(b.vdict[i] == i)) for i in b.vdict))
    b.clist = constraint_list((constraint((0, v_, i)) for i, v_ in enumerate(b.vlist)))
    b.p = parameter()
    b.pdict = parameter_dict(((i, parameter(i)) for i in b.vdict))
    b.plist = parameter_list((parameter(i) for i in range(len(b.vlist))))
    b.e = expression(b.v * b.p + 1)
    b.edict = expression_dict(((i, expression(b.vdict[i] * b.pdict[i] + 1)) for i in b.vdict))
    b.elist = expression_list((expression(v_ * b.plist[i] + 1) for i, v_ in enumerate(b.vlist)))
    self.assertIs(b.parent, None)
    bc = b.clone()
    self.assertIs(bc.parent, None)
    self.assertIsNot(b, bc)
    self.assertTrue(len(list(b.children())) > 0)
    self.assertEqual(len(list(b.children())), len(list(bc.children())))
    for c1, c2 in zip(b.children(), bc.children()):
        self.assertIs(c1.parent, b)
        self.assertIs(c2.parent, bc)
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
    self.assertEqual(len(list(b.components())), len(list(bc.components())))
    for c1, c2 in zip(b.components(), bc.components()):
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
        if hasattr(c1, 'expr'):
            self.assertIsNot(c1.expr, c2.expr)
            self.assertEqual(str(c1.expr), str(c2.expr))
            self.assertEqual(len(_collect_expr_components(c1.expr)), len(_collect_expr_components(c2.expr)))
            for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(), _collect_expr_components(c2.expr).values()):
                self.assertIsNot(subc1, subc2)
                self.assertEqual(subc1.name, subc2.name)
    bc_init = bc.clone()
    b.bc = bc
    self.assertIs(b.parent, None)
    self.assertIs(bc.parent, b)
    bcc = b.clone()
    self.assertIsNot(b, bcc)
    self.assertEqual(len(list(b.children())), len(list(bcc.children())))
    for c1, c2 in zip(b.children(), bcc.children()):
        self.assertIs(c1.parent, b)
        self.assertIs(c2.parent, bcc)
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
    self.assertEqual(len(list(b.components())), len(list(bcc.components())))
    self.assertTrue(hasattr(bcc, 'bc'))
    for c1, c2 in zip(b.components(), bcc.components()):
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
        if hasattr(c1, 'expr'):
            self.assertIsNot(c1.expr, c2.expr)
            self.assertEqual(str(c1.expr), str(c2.expr))
            self.assertEqual(len(_collect_expr_components(c1.expr)), len(_collect_expr_components(c2.expr)))
            for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(), _collect_expr_components(c2.expr).values()):
                self.assertIsNot(subc1, subc2)
                self.assertEqual(subc1.name, subc2.name)
    sub_bc = b.bc.clone()
    self.assertIs(sub_bc.parent, None)
    self.assertIs(bc_init.parent, None)
    self.assertIs(bc.parent, b)
    self.assertIs(b.parent, None)
    self.assertIsNot(bc_init, sub_bc)
    self.assertIsNot(bc, sub_bc)
    self.assertEqual(len(list(bc_init.children())), len(list(sub_bc.children())))
    for c1, c2 in zip(bc_init.children(), sub_bc.children()):
        self.assertIs(c1.parent, bc_init)
        self.assertIs(c2.parent, sub_bc)
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
    self.assertEqual(len(list(bc_init.components())), len(list(sub_bc.components())))
    for c1, c2 in zip(bc_init.components(), sub_bc.components()):
        self.assertIsNot(c1, c2)
        self.assertEqual(c1.name, c2.name)
        if hasattr(c1, 'expr'):
            self.assertIsNot(c1.expr, c2.expr)
            self.assertEqual(str(c1.expr), str(c2.expr))
            self.assertEqual(len(_collect_expr_components(c1.expr)), len(_collect_expr_components(c2.expr)))
            for subc1, subc2 in zip(_collect_expr_components(c1.expr).values(), _collect_expr_components(c2.expr).values()):
                self.assertIsNot(subc1, subc2)
                self.assertEqual(subc1.name, subc2.name)