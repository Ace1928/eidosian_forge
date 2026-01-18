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
def test_external_function_str_args(self):
    m = ConcreteModel()
    m.x = Var()
    m.e = ExternalFunction(library='tmp', function='test')
    m.o = Objective(expr=m.e(m.x, 'str'))
    OUT = io.StringIO(newline='\r\n')
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertIn("Writing NL file containing string arguments to a text output stream with line endings other than '\\n' ", LOG.getvalue())
    with TempfileManager:
        fname = TempfileManager.create_tempfile()
        with open(fname, 'w') as OUT:
            with LoggingIntercept() as LOG:
                nl_writer.NLWriter().write(m, OUT)
    if os.linesep == '\n':
        self.assertEqual(LOG.getvalue(), '')
    else:
        self.assertIn("Writing NL file containing string arguments to a text output stream with line endings other than '\\n' ", LOG.getvalue())
    r, w = os.pipe()
    try:
        OUT = os.fdopen(w, 'w')
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        if os.linesep == '\n':
            self.assertEqual(LOG.getvalue(), '')
        else:
            self.assertIn('Writing NL file containing string arguments to a text output stream that does not support tell()', LOG.getvalue())
    finally:
        OUT.close()
        os.close(r)