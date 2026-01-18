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
def test_load_solution(self):
    sm = SymbolMap()
    m = block()
    sm.addSymbol(m, 'm')
    m.v = variable()
    sm.addSymbol(m.v, 'v')
    m.c = constraint()
    sm.addSymbol(m.c, 'c')
    m.o = objective()
    sm.addSymbol(m.o, 'o')
    m.vsuffix = suffix(direction=suffix.IMPORT)
    m.osuffix = suffix(direction=suffix.IMPORT)
    m.csuffix = suffix(direction=suffix.IMPORT)
    m.msuffix = suffix(direction=suffix.IMPORT)
    soln = Solution()
    soln.symbol_map = sm
    soln.variable['v'] = {'Value': 1.0, 'vsuffix': 'v'}
    soln.variable['ONE_VAR_CONSTANT'] = None
    soln.constraint['c'] = {'csuffix': 'c'}
    soln.constraint['ONE_VAR_CONSTANT'] = None
    soln.objective['o'] = {'osuffix': 'o'}
    soln.problem['msuffix'] = 'm'
    m.load_solution(soln)
    self.assertEqual(m.v.value, 1.0)
    self.assertEqual(m.csuffix[m.c], 'c')
    self.assertEqual(m.osuffix[m.o], 'o')
    self.assertEqual(m.msuffix[m], 'm')
    soln.variable['vv'] = {'Value': 1.0, 'vsuffix': 'v'}
    with self.assertRaises(KeyError):
        m.load_solution(soln)
    del soln.variable['vv']
    soln.constraint['cc'] = {'csuffix': 'c'}
    with self.assertRaises(KeyError):
        m.load_solution(soln)
    del soln.constraint['cc']
    soln.objective['oo'] = {'osuffix': 'o'}
    with self.assertRaises(KeyError):
        m.load_solution(soln)
    del soln.objective['oo']
    m.v.fix()
    with self.assertRaises(ValueError):
        m.load_solution(soln, allow_consistent_values_for_fixed_vars=False)
    m.v.fix(1.1)
    m.load_solution(soln, allow_consistent_values_for_fixed_vars=True, comparison_tolerance_for_fixed_vars=0.5)
    m.v.fix(1.1)
    with self.assertRaises(ValueError):
        m.load_solution(soln, allow_consistent_values_for_fixed_vars=True, comparison_tolerance_for_fixed_vars=0.05)
    del soln.variable['v']
    m.v.free()
    m.v.value = None
    m.load_solution(soln)
    self.assertEqual(m.v.stale, True)
    self.assertEqual(m.v.value, None)
    soln.default_variable_value = 1.0
    m.load_solution(soln)
    self.assertEqual(m.v.stale, False)
    self.assertEqual(m.v.value, 1.0)
    m.v.fix(1.0)
    with self.assertRaises(ValueError):
        m.load_solution(soln, allow_consistent_values_for_fixed_vars=False)
    m.v.fix(1.1)
    m.load_solution(soln, allow_consistent_values_for_fixed_vars=True, comparison_tolerance_for_fixed_vars=0.5)
    m.v.fix(1.1)
    with self.assertRaises(ValueError):
        m.load_solution(soln, allow_consistent_values_for_fixed_vars=True, comparison_tolerance_for_fixed_vars=0.05)