import logging
from io import StringIO
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.trustregion.examples import example1, example2
from pyomo.environ import SolverFactory
def test_example1(self):
    log_OUTPUT = StringIO()
    print_OUTPUT = StringIO()
    sys.stdout = print_OUTPUT
    with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
        example1.main()
    sys.stdout = sys.__stdout__
    self.assertIn('Iteration 0', log_OUTPUT.getvalue())
    self.assertIn('Iteration 4', log_OUTPUT.getvalue())
    self.assertNotIn('Iteration 5', log_OUTPUT.getvalue())
    self.assertIn('theta-type step', log_OUTPUT.getvalue())
    self.assertNotIn('f-type step', log_OUTPUT.getvalue())
    self.assertNotIn('EXIT: Optimal solution found.', log_OUTPUT.getvalue())
    self.assertNotIn('None :   True : 0.2770447887637415', log_OUTPUT.getvalue())
    self.assertIn('Iteration 0', print_OUTPUT.getvalue())
    self.assertIn('Iteration 4', print_OUTPUT.getvalue())
    self.assertNotIn('Iteration 5', print_OUTPUT.getvalue())
    self.assertIn('theta-type step', print_OUTPUT.getvalue())
    self.assertNotIn('f-type step', print_OUTPUT.getvalue())
    self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
    self.assertIn('None :   True : 0.2770447887637415', print_OUTPUT.getvalue())