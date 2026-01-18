import os.path
from pyomo.common.fileutils import this_file_dir, import_file
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
from pyomo.opt import TerminationCondition
from io import StringIO
import logging
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver as cyipopt_solver
import cyipopt as cyipopt_core
def test_external_grey_box_react_example_maximize_with_additional_pyomo_variables(self):
    ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react_example', 'maximize_cb_ratio_residuals.py'))
    m = ex.maximize_cb_ratio_residuals_with_pyomo_variables()
    self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.26541996, places=3)
    self.assertAlmostEqual(pyo.value(m.reactor.inputs['cb']), 1071.7410089, places=2)
    self.assertAlmostEqual(pyo.value(m.cb_ratio), 0.15190409266, places=3)