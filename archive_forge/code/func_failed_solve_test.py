import os
import types
import pyomo.common.unittest as unittest
from pyomo.solvers.tests.models.base import all_models
from pyomo.solvers.tests.testcases import generate_scenarios
from pyomo.common.log import LoggingIntercept
from io import StringIO
def failed_solve_test(self):
    model_class = test_case.model()
    model_class.generate_model(test_case.testcase.import_suffixes)
    model_class.warmstart_model()
    load_solutions = True
    symbolic_labels = False
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.solvers'):
        with LoggingIntercept(out, 'pyomo.opt'):
            opt, results = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    model_class.post_solve_test_validation(self, results)
    if len(results.solution) == 0:
        self.assertIn('No solution is available', out.getvalue())
    else:
        self.assertEqual(len(results.solution), 1)