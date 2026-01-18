from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def test_deactivated_disjunct(self):
    m = self.make_model()
    self.add_fourth_disjunct(m)
    m.d4.deactivate()
    mbm = TransformationFactory('gdp.mbigm')
    mbm.apply_to(m, bigM=self.get_Ms(m), reduce_bound_constraints=False)
    self.assertIsNone(m.d4.transformation_block)
    self.check_linear_func_constraints(m, mbm)
    self.check_all_untightened_bounds_constraints(m, mbm)