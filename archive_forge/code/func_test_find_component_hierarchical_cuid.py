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
def test_find_component_hierarchical_cuid(self):
    b1 = Block(concrete=True)
    b1.b2 = Block()
    b1.b2.v1 = Var()
    b1.b2.v2 = Var([1, 2])
    cuid1 = ComponentUID('b2.v1')
    cuid2 = ComponentUID('b2.v2[2]')
    self.assertIs(b1.find_component(cuid1), b1.b2.v1)
    self.assertIs(b1.find_component(cuid2), b1.b2.v2[2])