import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_const_obj(self):
    m = ConcreteModel()
    m.x = Var()
    m.o = Objective(expr=5)
    with self.assertRaisesRegex(RuntimeError, 'constant objective'):
        SolverFactory('multistart').solve(m)