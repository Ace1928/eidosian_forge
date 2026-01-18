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
def test_collect_ctypes_small_block_storage(self):
    b = block()
    self.assertEqual(b.collect_ctypes(active=None), set())
    self.assertEqual(b.collect_ctypes(), set())
    self.assertEqual(b.collect_ctypes(active=True), set())
    b.x = variable()
    self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
    self.assertEqual(b.collect_ctypes(), set([IVariable]))
    self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
    b.y = constraint()
    self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
    self.assertEqual(b.collect_ctypes(), set([IVariable, IConstraint]))
    self.assertEqual(b.collect_ctypes(active=True), set([IVariable, IConstraint]))
    b.y.deactivate()
    self.assertEqual(b.collect_ctypes(active=None), set([IVariable, IConstraint]))
    self.assertEqual(b.collect_ctypes(), set([IVariable]))
    self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
    B = block()
    B.b = b
    self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
    self.assertEqual(B.collect_ctypes(descend_into=False), set([IBlock]))
    self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IBlock]))
    self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
    self.assertEqual(B.collect_ctypes(), set([IBlock, IVariable]))
    self.assertEqual(B.collect_ctypes(active=True), set([IBlock, IVariable]))
    b.deactivate()
    self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock]))
    self.assertEqual(B.collect_ctypes(descend_into=False), set([]))
    self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([]))
    self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
    self.assertEqual(B.collect_ctypes(), set([]))
    self.assertEqual(B.collect_ctypes(active=True), set([]))
    B.x = variable()
    self.assertEqual(B.collect_ctypes(descend_into=False, active=None), set([IBlock, IVariable]))
    self.assertEqual(B.collect_ctypes(descend_into=False), set([IVariable]))
    self.assertEqual(B.collect_ctypes(descend_into=False, active=True), set([IVariable]))
    self.assertEqual(B.collect_ctypes(active=None), set([IBlock, IVariable, IConstraint]))
    self.assertEqual(B.collect_ctypes(), set([IVariable]))
    self.assertEqual(B.collect_ctypes(active=True), set([IVariable]))
    del b.y
    self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
    self.assertEqual(b.collect_ctypes(), set([]))
    self.assertEqual(b.collect_ctypes(active=True), set([]))
    b.activate()
    self.assertEqual(b.collect_ctypes(active=None), set([IVariable]))
    self.assertEqual(b.collect_ctypes(), set([IVariable]))
    self.assertEqual(b.collect_ctypes(active=True), set([IVariable]))
    del b.x
    self.assertEqual(b.collect_ctypes(), set())