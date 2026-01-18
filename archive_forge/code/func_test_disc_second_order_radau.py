import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_second_order_radau(self):
    m = self.m.clone()
    m.dv1dt2 = DerivativeVar(m.v1, wrt=(m.t, m.t))
    disc = TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=2, ncp=2)
    self.assertTrue(hasattr(m, 'dv1dt2_disc_eq'))
    self.assertTrue(len(m.dv1dt2_disc_eq) == 4)
    self.assertTrue(len(m.v1) == 5)
    self.assertTrue(hasattr(m, '_pyomo_dae_reclassified_derivativevars'))
    self.assertTrue(m.dv1 in m._pyomo_dae_reclassified_derivativevars)
    self.assertTrue(m.dv1dt2 in m._pyomo_dae_reclassified_derivativevars)
    repn_baseline = {id(m.dv1dt2[5.0]): 1, id(m.v1[0]): -0.24, id(m.v1[1.666667]): 0.36, id(m.v1[5.0]): -0.12}
    repn = generate_standard_repn(m.dv1dt2_disc_eq[5.0].body)
    repn_gen = repn_to_rounded_dict(repn, 5)
    self.assertEqual(repn_baseline, repn_gen)
    repn_baseline = {id(m.dv1dt2[10]): 1, id(m.v1[5.0]): -0.24, id(m.v1[6.666667]): 0.36, id(m.v1[10]): -0.12}
    repn = generate_standard_repn(m.dv1dt2_disc_eq[10.0].body)
    repn_gen = repn_to_rounded_dict(repn, 5)
    self.assertEqual(repn_baseline, repn_gen)