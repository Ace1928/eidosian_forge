import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_initialized_continuous_set(self):
    m = ConcreteModel()
    m.t = ContinuousSet(initialize=[0, 1, 2, 3, 4])
    m.v = Var(m.t)
    m.dv = DerivativeVar(m.v)
    log_out = StringIO()
    with LoggingIntercept(log_out, 'pyomo.dae'):
        TransformationFactory('dae.finite_difference').apply_to(m, nfe=2)
    self.assertIn('More finite elements', log_out.getvalue())