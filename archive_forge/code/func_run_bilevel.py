import os
from os.path import abspath, dirname, normpath, join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
from pyomo.scripting.util import cleanup
import pyomo.scripting.pyomo_main as main
def run_bilevel(self, *args, **kwds):
    kwds['solver'] = 'cplex'
    CommonTests.run_bilevel(self, *args, **kwds)