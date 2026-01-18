import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_multi_index2(self):
    m = self.m.clone()
    m.t2 = ContinuousSet(bounds=(0, 5))
    m.v2 = Var(m.t, m.t2)
    m.dv2dt = DerivativeVar(m.v2, wrt=m.t)
    m.dv2dt2 = DerivativeVar(m.v2, wrt=m.t2)
    disc = TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=2)
    self.assertTrue(hasattr(m, 'dv2dt_disc_eq'))
    self.assertTrue(hasattr(m, 'dv2dt2_disc_eq'))
    self.assertEqual(len(m.dv2dt_disc_eq), 6)
    self.assertEqual(len(m.dv2dt2_disc_eq), 6)
    self.assertEqual(len(m.v2), 9)
    expected_t_disc_points = [0, 5.0, 10]
    expected_t2_disc_points = [0, 2.5, 5]
    for idx, val in enumerate(list(m.t)):
        self.assertAlmostEqual(val, expected_t_disc_points[idx])
    for idx, val in enumerate(list(m.t2)):
        self.assertAlmostEqual(val, expected_t2_disc_points[idx])