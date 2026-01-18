import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def test_external_expression_partial_fixed(self):
    m = self._external_model()
    m.x.fix()
    baseline_fname, test_fname = self._get_fnames()
    m.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True, 'column_order': True})
    self._compare_nl_baseline(baseline_fname, test_fname)