import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
def test_cov5(self):
    M = self._setup()

    def f(model):
        raise IOError('cov5 error')
    try:
        M.cc1 = Complementarity(rule=f)
        self.fail('Expected an IOError')
    except IOError:
        pass

    def f(model, i):
        raise IOError('cov5 error')
    try:
        M.cc2 = Complementarity([0, 1], rule=f)
        self.fail('Expected an IOError')
    except IOError:
        pass