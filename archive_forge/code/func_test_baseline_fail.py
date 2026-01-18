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
def test_baseline_fail(self):
    with self.assertRaises(self.failureException):
        with capture_output() as OUT:
            self.compare_baseline(fail_ref, baseline)
    self.assertEqual(OUT.getvalue(), f'---------------------------------\nBASELINE FILE\n---------------------------------\n{baseline}\n=================================\n---------------------------------\nTEST OUTPUT FILE\n---------------------------------\n{fail_ref}\n')