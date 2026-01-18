import re
import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
from pyomo.common.log import LoggingIntercept
from pyomo.environ import SolverFactory
from pyomo.scripting.driver_help import help_solvers, help_transformations
from pyomo.scripting.pyomo_main import main
def test_help_solvers(self):
    with capture_output() as OUT:
        help_solvers()
    OUT = OUT.getvalue()
    self.assertTrue(re.search('Pyomo Solvers and Solver Managers', OUT))
    self.assertTrue(re.search('Serial Solver', OUT))
    self.assertTrue(re.search('\\n   \\*asl ', OUT))
    self.assertTrue(re.search('\\n   \\+mindtpy ', OUT))
    for solver in ('ipopt', 'cbc', 'glpk'):
        s = SolverFactory(solver)
        if s.available():
            self.assertTrue(re.search('\\n   \\+%s ' % solver, OUT), "'   +%s' not found in help --solvers" % solver)
        else:
            self.assertTrue(re.search('\\n    %s ' % solver, OUT), "'    %s' not found in help --solvers" % solver)
    for solver in ('baron',):
        s = SolverFactory(solver)
        if s.license_is_valid():
            self.assertTrue(re.search('\\n   \\+%s ' % solver, OUT), "'   +%s' not found in help --solvers" % solver)
        elif s.available():
            self.assertTrue(re.search('\\n   \\-%s ' % solver, OUT), "'   -%s' not found in help --solvers" % solver)
        else:
            self.assertTrue(re.search('\\n    %s ' % solver, OUT), "'    %s' not found in help --solvers" % solver)