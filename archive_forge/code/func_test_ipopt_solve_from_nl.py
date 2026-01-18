import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.core import (
def test_ipopt_solve_from_nl(self):
    _log = TempfileManager.create_tempfile('.test_ipopt.log')
    results = self.ipopt.solve(join(currdir, 'sisser.pyomo.nl'), logfile=_log, suffixes=['.*'])
    results.Solution(0).Message = 'Ipopt'
    results.Solver.Message = 'Ipopt'
    _out = TempfileManager.create_tempfile('.test_ipopt.txt')
    results.write(filename=_out, times=False, format='json')
    self.compare_json(_out, join(currdir, 'test_solve_from_nl.baseline'))