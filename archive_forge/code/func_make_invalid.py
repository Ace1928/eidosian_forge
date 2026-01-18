import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def make_invalid(m):
    m.I = RangeSet(3)
    m.x = Var(m.I)
    m.c = Constraint(expr=sum((m.x[i] for i in m.I)) >= 0)