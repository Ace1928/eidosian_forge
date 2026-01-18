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
def init_block(b):
    b.c = Block([1, 2], rule=def_var)
    b.e = Disjunct([1, 2], rule=def_var)
    b.b = Block(rule=def_var)
    b.d = Disjunct(rule=def_var)