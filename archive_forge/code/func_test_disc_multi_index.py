import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_multi_index(self):
    m = self.m.clone()
    m.v2 = Var(m.t, m.s)
    m.dv2 = DerivativeVar(m.v2)
    disc = TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=5)
    self.assertTrue(hasattr(m, 'dv1_disc_eq'))
    self.assertTrue(hasattr(m, 'dv2_disc_eq'))
    self.assertEqual(len(m.dv2_disc_eq), 15)
    self.assertEqual(len(m.v2), 18)
    expected_disc_points = [0, 2.0, 4.0, 6.0, 8.0, 10]
    disc_info = m.t.get_discretization_info()
    self.assertEqual(disc_info['scheme'], 'BACKWARD Difference')
    for idx, val in enumerate(list(m.t)):
        self.assertAlmostEqual(val, expected_disc_points[idx])