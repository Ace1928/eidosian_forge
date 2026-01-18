import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_lookup_radau_collocation_points(self):
    colloc_numpy_avail = pyomo.dae.plugins.colloc.numpy_available
    pyomo.dae.plugins.colloc.numpy_available = False
    m = self.m.clone()
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=5, ncp=3)
    self.assertTrue(hasattr(m, 'dv1_disc_eq'))
    self.assertTrue(len(m.dv1_disc_eq) == 15)
    self.assertTrue(len(m.v1) == 16)
    expected_tau_points = [0.0, 0.1550510257216822, 0.6449489742783179, 1.0]
    expected_disc_points = [0, 0.310102, 1.289898, 2.0, 2.310102, 3.289898, 4.0, 4.310102, 5.289898, 6.0, 6.310102, 7.289898, 8.0, 8.310102, 9.289898, 10]
    disc_info = m.t.get_discretization_info()
    self.assertTrue(disc_info['scheme'] == 'LAGRANGE-RADAU')
    for idx, val in enumerate(disc_info['tau_points']):
        self.assertAlmostEqual(val, expected_tau_points[idx])
    for idx, val in enumerate(list(m.t)):
        self.assertAlmostEqual(val, expected_disc_points[idx])
    m = self.m.clone()
    with self.assertRaises(ValueError):
        disc = TransformationFactory('dae.collocation')
        disc.apply_to(m, ncp=15, scheme='LAGRANGE-RADAU')
    pyomo.dae.plugins.colloc.numpy_available = colloc_numpy_avail