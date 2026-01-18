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
@unittest.skipUnless(gurobi_available, 'Gurobi is not available')
def test_complain_for_unrecognized_Suffix(self):
    m = self.make_infeasible_disjunct_model()
    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[1].HiThere = Suffix(direction=Suffix.LOCAL)
    out = StringIO()
    with self.assertRaisesRegex(GDP_Error, "Found active Suffix 'disjunction_disjuncts\\[1\\].HiThere' on Disjunct 'disjunction_disjuncts\\[1\\]'. The multiple bigM transformation does not support this Suffix."):
        TransformationFactory('gdp.mbigm').apply_to(m, reduce_bound_constraints=False)