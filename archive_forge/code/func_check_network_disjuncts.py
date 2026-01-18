import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_network_disjuncts(self, minimize, transformation, **kwds):
    m = models.makeExpandedNetworkDisjunction(minimize=minimize)
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwds)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    if minimize:
        self.assertAlmostEqual(value(m.dest.x), 0.42)
    else:
        self.assertAlmostEqual(value(m.dest.x), 0.84)