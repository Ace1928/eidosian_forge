import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.core import (
def test_ipopt_solve_from_instance_OF_options(self):
    with self.assertRaises(ValueError):
        self.ipopt.solve(self.sisser_instance, suffixes=['.*'], options={'OF_mu_init': 0.1, 'option_file_name': 'junk.opt'})
    _cwd = os.getcwd()
    tmpdir = TempfileManager.create_tempdir()
    try:
        os.chdir(tmpdir)
        open(join(tmpdir, 'ipopt.opt'), 'w').close()
        with LoggingIntercept() as LOG:
            results = self.ipopt.solve(self.sisser_instance, suffixes=['.*'], options={'OF_mu_init': 0.1})
        self.assertRegex(LOG.getvalue().replace('\n', ' '), "A file named (.*) exists in the current working directory, but Ipopt options file options \\(i.e., options that start with 'OF_'\\) were provided. The options file \\1 will be ignored.")
    finally:
        os.chdir(_cwd)
    self.sisser_instance.solutions.store_to(results)
    results.Solution(0).Message = 'Ipopt'
    results.Solver.Message = 'Ipopt'
    _out = TempfileManager.create_tempfile('.test_ipopt.txt')
    results.write(filename=_out, times=False, format='json')
    self.compare_json(_out, join(currdir, 'test_solve_from_instance.baseline'))