import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory, pyomo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.repn import generate_standard_repn
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_disc_second_order_1cp(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 1))
    m.t2 = ContinuousSet(bounds=(0, 10))
    m.v = Var(m.t, m.t2)
    m.dv = DerivativeVar(m.v, wrt=(m.t, m.t2))
    TransformationFactory('dae.collocation').apply_to(m, nfe=2, ncp=1)
    self.assertTrue(hasattr(m, 'dv_disc_eq'))
    self.assertTrue(len(m.dv_disc_eq) == 4)
    self.assertTrue(len(m.v) == 9)