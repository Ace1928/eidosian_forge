import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
def test_bound_function_require_fork(self):
    if multiprocessing.get_start_method() == 'fork':
        self.bound_function_require_fork()
        return
    with self.assertRaisesRegex(unittest.SkipTest, 'timeout requires unavailable fork interface'):
        self.bound_function_require_fork()