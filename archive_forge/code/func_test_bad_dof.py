import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.core import (
def test_bad_dof(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.c = ConstraintList()
    m.c.add(m.x + m.y == 1)
    m.c.add(m.x - m.y == 0)
    m.c.add(2 * m.x - 3 * m.y == 1)
    res = self.ipopt.solve(m)
    self.assertEqual(str(res.solver.status), 'warning')
    self.assertEqual(str(res.solver.termination_condition), 'other')
    self.assertTrue('Too few degrees of freedom' in res.solver.message)