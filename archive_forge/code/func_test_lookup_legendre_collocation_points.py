import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_lookup_legendre_collocation_points(self):
    colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available
    pyomo.dae.plugins.colloc.numpy_available = False
    m = self.m.clone()
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=5, ncp=3, scheme='LAGRANGE-LEGENDRE')
    self.assertTrue(hasattr(m, 'dv1_disc_eq'))
    self.assertTrue(len(m.dv1_disc_eq) == 15)
    self.assertTrue(len(m.v1) == 21)
    expected_tau_points = [0.0, 0.11270166537925834, 0.4999999999999999, 0.8872983346207423]
    expected_disc_points = [0, 0.225403, 1.0, 1.774597, 2.0, 2.225403, 3.0, 3.774597, 4.0, 4.225403, 5.0, 5.774597, 6.0, 6.225403, 7.0, 7.774597, 8.0, 8.225403, 9.0, 9.774597, 10]
    disc_info = m.t.get_discretization_info()
    self.assertTrue(disc_info['scheme'] == 'LAGRANGE-LEGENDRE')
    for idx, val in enumerate(disc_info['tau_points']):
        self.assertAlmostEqual(val, expected_tau_points[idx])
    for idx, val in enumerate(list(m.t)):
        self.assertAlmostEqual(val, expected_disc_points[idx])
    m = self.m.clone()
    with self.assertRaises(ValueError):
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, ncp=15, scheme='LAGRANGE-LEGENDRE')
    pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail