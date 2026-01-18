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
def test_customblock_setattr(self):
    b = _MyBlockBase()
    self.assertIs(b.b.parent, b)
    self.assertIs(b.b.v.parent, b.b)
    with self.assertRaises(ValueError):
        b.b = b.b.v
    self.assertIs(b.b.parent, b)
    self.assertIs(b.b.v.parent, b.b)
    c = b.b
    self.assertIs(c.parent, b)
    b.b = c
    self.assertIs(c.parent, b)
    assert not hasattr(b, 'g')
    with self.assertRaises(ValueError):
        b.g = b.b
    self.assertIs(b.b.parent, b)
    b.g = 1
    with self.assertRaises(ValueError):
        b.g = b.b
    self.assertEqual(b.g, 1)
    self.assertIs(b.b.parent, b)
    b.b = block()
    self.assertIs(c.parent, None)
    self.assertIs(b.b.parent, b)