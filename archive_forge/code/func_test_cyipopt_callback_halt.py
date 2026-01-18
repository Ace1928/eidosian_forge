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
def test_cyipopt_callback_halt(self):
    ex = import_file(os.path.join(example_dir, 'callback', 'cyipopt_callback_halt.py'))
    status = ex.main()
    self.assertEqual(status.solver.termination_condition, TerminationCondition.userInterrupt)