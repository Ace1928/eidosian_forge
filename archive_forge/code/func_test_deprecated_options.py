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
def test_deprecated_options(self):
    m = ConcreteModel()

    def b_rule(b, a=None):
        b.p = Param(initialize=a)
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        m.b = Block(rule=b_rule, options={'a': 5})
    self.assertIn("The Block 'options=' keyword is deprecated.", OUTPUT.getvalue())
    self.assertEqual(value(m.b.p), 5)
    m = ConcreteModel()

    def b_rule(b, i, **kwds):
        b.p = Param(initialize=kwds.get('a', {}).get(i, 0))
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        m.b = Block([1, 2, 3], rule=b_rule, options={'a': {1: 5, 2: 10}})
    self.assertIn("The Block 'options=' keyword is deprecated.", OUTPUT.getvalue())
    self.assertEqual(value(m.b[1].p), 5)
    self.assertEqual(value(m.b[2].p), 10)
    self.assertEqual(value(m.b[3].p), 0)