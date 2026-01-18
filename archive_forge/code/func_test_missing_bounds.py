import logging
from itertools import product
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.multistart.high_conf_stop import should_stop
from pyomo.contrib.multistart.reinit import strategies
from pyomo.environ import (
def test_missing_bounds(self):
    m = ConcreteModel()
    m.x = Var(domain=NonNegativeReals)
    m.obj = Objective(expr=m.x)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.multistart', logging.WARNING):
        SolverFactory('multistart').solve(m)
        self.assertIn('Skipping reinitialization of unbounded variable x with bounds (0, None).', output.getvalue().strip())