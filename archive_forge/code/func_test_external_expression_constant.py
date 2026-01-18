import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def test_external_expression_constant(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    m = ConcreteModel()
    m.y = Var(initialize=4, bounds=(0, None))
    m.hypot = ExternalFunction(library=DLL, function='gsl_hypot')
    m.o = Objective(expr=m.hypot(3, m.y))
    self.assertAlmostEqual(value(m.o), 5.0, 7)
    baseline_fname, test_fname = self._get_fnames()
    m.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True})
    self._compare_nl_baseline(baseline_fname, test_fname)