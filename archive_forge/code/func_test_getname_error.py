import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
def test_getname_error(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.v = Var()
    m.c = Block()
    self.assertRaises(RuntimeError, m.b.v.getname, fully_qualified=True, relative_to=m.c)