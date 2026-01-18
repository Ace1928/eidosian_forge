import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
@unittest.skipUnless(find_GSL(), 'Could not find the AMPL GSL library')
def test_identify_external_functions(self):
    m = self._make_model_with_external_functions()
    m._con = pyo.Constraint(expr=2 * m.fermi(m.bessel(m.v1 ** 2) + 0.1) == 1.0)
    gsl = find_GSL()
    fcns = list(identify_external_functions(m.con2.expr))
    self.assertEqual(len(fcns), 1)
    self.assertEqual(fcns[0]._fcn._library, gsl)
    self.assertEqual(fcns[0]._fcn._function, 'gsl_sf_fermi_dirac_m1')
    fcns = list(identify_external_functions(m.con3.expr))
    fcn_data = set(((fcn._fcn._library, fcn._fcn._function) for fcn in fcns))
    self.assertEqual(len(fcns), 2)
    pred_fcn_data = {(gsl, 'gsl_sf_bessel_J0')}
    self.assertEqual(fcn_data, pred_fcn_data)
    fcns = list(identify_external_functions(m._con.expr))
    fcn_data = set(((fcn._fcn._library, fcn._fcn._function) for fcn in fcns))
    self.assertEqual(len(fcns), 2)
    pred_fcn_data = {(gsl, 'gsl_sf_bessel_J0'), (gsl, 'gsl_sf_fermi_dirac_m1')}
    self.assertEqual(fcn_data, pred_fcn_data)