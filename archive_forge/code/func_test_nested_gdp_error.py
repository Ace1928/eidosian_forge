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
def test_nested_gdp_error(self):
    m = self.make_model()
    m.d1.disjunction = Disjunction(expr=[m.x1 >= 5, m.x1 <= 4])
    with self.assertRaisesRegex(GDP_Error, "Found nested Disjunction 'd1.disjunction'. The multiple bigm transformation does not support nested GDPs. Please flatten the model before calling the transformation"):
        TransformationFactory('gdp.mbigm').apply_to(m)