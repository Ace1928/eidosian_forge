import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
def test_external_function_with_potential_name_collision(self):
    m = self._make_model_with_external_functions()
    m.b = pyo.Block()
    m.b._gsl_sf_bessel_J0 = pyo.Var()
    m.b.con = pyo.Constraint(expr=m.b._gsl_sf_bessel_J0 == m.bessel(m.v1))
    add_local_external_functions(m.b)
    self.assertTrue(isinstance(m.b._gsl_sf_bessel_J0, pyo.Var))
    ex_fcns = list(m.b.component_objects(pyo.ExternalFunction))
    self.assertEqual(len(ex_fcns), 1)
    fcn = ex_fcns[0]
    self.assertEqual(fcn._function, 'gsl_sf_bessel_J0')