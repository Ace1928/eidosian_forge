import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_log_timing(self):
    m = ConcreteModel()
    m.x = Var(range(6))
    m.x[0].domain = pyo.Binary
    m.x[1].domain = pyo.Integers
    m.x[2].domain = pyo.Integers
    m.p = Param(initialize=5, mutable=True)
    m.o1 = Objective([1, 2], rule=lambda m, i: 1)
    m.o2 = Objective(expr=m.x[1] * m.x[2])
    m.c1 = Constraint([1, 2], rule=lambda m, i: sum(m.x.values()) == 1)
    m.c2 = Constraint(expr=m.p * m.x[1] ** 2 + m.x[2] ** 3 <= 100)
    self.maxDiff = None
    OUT = io.StringIO()
    with capture_output() as LOG:
        with report_timing(level=logging.DEBUG):
            nl_writer.NLWriter().write(m, OUT)
    self.assertEqual('      [+   #.##] Initialized column order\n      [+   #.##] Collected suffixes\n      [+   #.##] Objective o1\n      [+   #.##] Objective o2\n      [+   #.##] Constraint c1\n      [+   #.##] Constraint c2\n      [+   #.##] Categorized model variables: 14 nnz\n      [+   #.##] Set row / column ordering: 6 var [3, 1, 2 R/B/Z], 3 con [2, 1 L/NL]\n      [+   #.##] Generated row/col labels & comments\n      [+   #.##] Wrote NL stream\n      [    #.##] Generated NL representation\n', re.sub('\\d\\.\\d\\d\\]', '#.##]', LOG.getvalue()))