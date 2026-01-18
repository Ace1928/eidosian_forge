import os
from os.path import abspath, dirname
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (
from pyomo.core.base import TransformationFactory
from pyomo.core.expr import log
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination
from io import StringIO
import logging
import random
@staticmethod
def makeModel():
    """
        This is a single-level reformulation of a bilevel model.
        We project out the dual variables to recover the reformulation in
        the original space.
        """
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.y = Var(domain=NonNegativeReals)
    m.lamb = Var([1, 2], domain=NonNegativeReals)
    m.M = Param([1, 2], mutable=True, default=100)
    m.u = Var([1, 2], domain=Binary)
    m.primal1 = Constraint(expr=m.x - 0.01 * m.y <= 1)
    m.dual1 = Constraint(expr=1 - m.lamb[1] - 0.01 * m.lamb[2] == 0)

    @m.Constraint([1, 2])
    def bound_lambdas(m, i):
        return m.lamb[i] <= m.u[i] * m.M[i]
    m.bound_y = Constraint(expr=m.y <= 1000 * (1 - m.u[1]))
    m.dual2 = Constraint(expr=-m.x + 0.01 * m.y + 1 <= (1 - m.u[2]) * 1000)
    return m