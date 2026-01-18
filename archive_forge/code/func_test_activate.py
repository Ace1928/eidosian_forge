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
def test_activate(self):
    b = block()
    self.assertEqual(b.active, True)
    b.deactivate()
    self.assertEqual(b.active, False)
    c = constraint()
    v = variable()
    self.assertEqual(c.active, True)
    c.deactivate()
    self.assertEqual(c.active, False)
    b.c = c
    b.v = v
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, False)
    del b.c
    c.activate()
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, False)
    b.c = c
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, False)
    bdict = block_dict()
    self.assertEqual(bdict.active, True)
    bdict.deactivate()
    self.assertEqual(bdict.active, False)
    bdict[None] = b
    self.assertEqual(bdict.active, False)
    del bdict[None]
    self.assertEqual(bdict.active, False)
    b.activate()
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, True)
    bdict[None] = b
    self.assertEqual(bdict.active, False)
    bdict.deactivate()
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, True)
    self.assertEqual(bdict.active, False)
    bdict.deactivate(shallow=False)
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, False)
    self.assertEqual(bdict.active, False)
    b.activate()
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, True)
    self.assertEqual(bdict.active, False)
    b.activate(shallow=False)
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, True)
    self.assertEqual(bdict.active, False)
    bdict.deactivate()
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, True)
    self.assertEqual(bdict.active, False)
    bdict.deactivate(shallow=False)
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, False)
    self.assertEqual(bdict.active, False)
    bdict.activate()
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, False)
    self.assertEqual(bdict.active, True)
    bdict.activate(shallow=False)
    self.assertEqual(c.active, True)
    self.assertEqual(b.active, True)
    self.assertEqual(bdict.active, True)
    bdict.deactivate(shallow=False)
    self.assertEqual(c.active, False)
    self.assertEqual(b.active, False)
    self.assertEqual(bdict.active, False)