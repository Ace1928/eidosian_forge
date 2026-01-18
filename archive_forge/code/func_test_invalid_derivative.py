import pyomo.common.unittest as unittest
from pyomo.environ import Var, Set, ConcreteModel, TransformationFactory
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from io import StringIO
from pyomo.common.log import LoggingIntercept
from os.path import abspath, dirname, normpath, join
def test_invalid_derivative(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.v = Var(m.t)
    m.dv = DerivativeVar(m.v, wrt=(m.t, m.t, m.t))
    with self.assertRaises(DAE_Error):
        TransformationFactory('dae.finite_difference').apply_to(m)