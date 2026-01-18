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
def test_external_grey_box_react_example_maximize_cb_outputs(self):
    ex = import_file(os.path.join(example_dir, 'external_grey_box', 'react_example', 'maximize_cb_outputs.py'))
    m = ex.maximize_cb_outputs()
    self.assertAlmostEqual(pyo.value(m.reactor.inputs['sv']), 1.34381, places=3)
    self.assertAlmostEqual(pyo.value(m.reactor.outputs['cb']), 1072.4372, places=2)