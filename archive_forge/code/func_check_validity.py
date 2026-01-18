import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_validity(self, body, lower, upper, TOL=0):
    if lower is not None:
        self.assertGreaterEqual(value(body), value(lower) - TOL)
    if upper is not None:
        self.assertLessEqual(value(body), value(upper) + TOL)