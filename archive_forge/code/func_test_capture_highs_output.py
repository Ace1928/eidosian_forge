import subprocess
import sys
import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.base import TerminationCondition
def test_capture_highs_output(self):
    model = ['import pyomo.environ as pe', 'm = pe.ConcreteModel()', 'm.x = pe.Var(domain=pe.NonNegativeReals)', 'm.y = pe.Var(domain=pe.NonNegativeReals)', 'm.obj = pe.Objective(expr=m.x + m.y, sense=pe.maximize)', 'm.c1 = pe.Constraint(expr=m.x <= 10)', 'm.c2 = pe.Constraint(expr=m.y <= 5)', 'from pyomo.contrib.appsi.solvers.highs import Highs', 'result = Highs().solve(m)', 'print(m.x.value, m.y.value)']
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        subprocess.run([sys.executable, '-c', ';'.join(model)])
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(OUT.getvalue(), '10.0 5.0\n')
    model[-2:-1] = ['opt = Highs()', 'opt.config.stream_solver = True', 'result = opt.solve(m)']
    with LoggingIntercept() as LOG, capture_output(capture_fd=True) as OUT:
        subprocess.run([sys.executable, '-c', ';'.join(model)])
    self.assertEqual(LOG.getvalue(), '')
    self.assertIn('Running HiGHS', OUT.getvalue())
    self.assertIn('HiGHS run time', OUT.getvalue())
    ref = '10.0 5.0\n'
    self.assertEqual(ref, OUT.getvalue()[-len(ref):])