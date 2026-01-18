from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.contrib.appsi.utils import collect_vars_and_named_exprs
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from typing import Callable
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
def test_basics_cmodel(self):
    self.basics_helper(cmodel.prep_for_repn, cmodel.PyomoExprTypes())