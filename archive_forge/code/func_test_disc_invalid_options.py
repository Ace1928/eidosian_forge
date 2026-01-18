import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_invalid_options(self):
    m = self.m.clone()
    with self.assertRaises(TypeError):
        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.s)
    with self.assertRaises(ValueError):
        TransformationFactory('dae.finite_difference').apply_to(m, nfe=-1)
    with self.assertRaises(ValueError):
        TransformationFactory('dae.finite_difference').apply_to(m, scheme='foo')
    with self.assertRaises(ValueError):
        TransformationFactory('dae.finite_difference').apply_to(m, foo=True)
    TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)
    with self.assertRaises(ValueError):
        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)
    m = self.m.clone()
    disc = TransformationFactory('dae.finite_difference')
    disc.apply_to(m)
    with self.assertRaises(ValueError):
        disc.apply_to(m)