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
def create_hull_model(self):
    m = ConcreteModel()
    m.p = Var([1, 2], bounds=(0, 10))
    m.time1 = Disjunction(expr=[m.p[1] >= 1, m.p[1] == 0])
    m.on = Disjunct()
    m.on.above_min = Constraint(expr=m.p[2] >= 1)
    m.on.ramping = Constraint(expr=m.p[2] - m.p[1] <= 3)
    m.on.on_before = Constraint(expr=m.p[1] >= 1)
    m.startup = Disjunct()
    m.startup.startup_limit = Constraint(expr=(1, m.p[2], 2))
    m.startup.off_before = Constraint(expr=m.p[1] == 0)
    m.off = Disjunct()
    m.off.off = Constraint(expr=m.p[2] == 0)
    m.time2 = Disjunction(expr=[m.on, m.startup, m.off])
    m.obj = Objective(expr=m.p[1] + m.p[2])
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disaggregatedVars = ComponentSet([hull.get_disaggregated_var(m.p[1], m.time1.disjuncts[0]), hull.get_disaggregated_var(m.p[1], m.time1.disjuncts[1]), hull.get_disaggregated_var(m.p[1], m.on), hull.get_disaggregated_var(m.p[2], m.on), hull.get_disaggregated_var(m.p[1], m.startup), hull.get_disaggregated_var(m.p[2], m.startup), hull.get_disaggregated_var(m.p[1], m.off), hull.get_disaggregated_var(m.p[2], m.off)])
    return (m, disaggregatedVars)