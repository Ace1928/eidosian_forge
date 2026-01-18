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
def test_inactive_behavior(self):
    b = _MyBlock()
    b.deactivate()
    self.assertNotEqual(len(list(pmo.preorder_traversal(b, active=None))), 0)
    self.assertEqual(len(list(pmo.preorder_traversal(b))), 0)
    self.assertEqual(len(list(pmo.preorder_traversal(b, active=True))), 0)

    def descend(x):
        return True
    self.assertNotEqual(len(list(pmo.preorder_traversal(b, active=None, descend=descend))), 0)
    self.assertEqual(len(list(pmo.preorder_traversal(b, descend=descend))), 0)
    self.assertEqual(len(list(pmo.preorder_traversal(b, active=True, descend=descend))), 0)

    def descend(x):
        descend.seen.append(x)
        return x.active
    descend.seen = []
    self.assertEqual(len(list(pmo.preorder_traversal(b, active=None, descend=descend))), 1)
    self.assertEqual(len(descend.seen), 1)
    self.assertIs(descend.seen[0], b)
    self.assertNotEqual(len(list(b.components(active=None))), 0)
    self.assertEqual(len(list(b.components())), 0)
    self.assertEqual(len(list(b.components(active=True))), 0)
    self.assertNotEqual(len(list(pmo.generate_names(b, active=None))), 0)
    self.assertEqual(len(list(pmo.generate_names(b))), 0)
    self.assertEqual(len(list(pmo.generate_names(b, active=True))), 0)