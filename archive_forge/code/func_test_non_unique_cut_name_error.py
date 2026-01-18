import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def test_non_unique_cut_name_error(self):
    m = models.twoSegments_SawayaGrossmann()
    self.assertRaisesRegex(GDP_Error, "cuts_name was specified as 'disj1', but this is already a component on the instance! Please specify a unique name.", TransformationFactory('gdp.cuttingplane').apply_to, m, cuts_name='disj1')