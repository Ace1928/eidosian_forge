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
def check_iteratively_adding_to_indexed_disjunction_on_block(self, transformation):
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var(bounds=(-100, 100))
    m.b.firstTerm = Disjunct([1, 2])
    m.b.firstTerm[1].cons = Constraint(expr=m.b.x == 0)
    m.b.firstTerm[2].cons = Constraint(expr=m.b.x == 2)
    m.b.secondTerm = Disjunct([1, 2])
    m.b.secondTerm[1].cons = Constraint(expr=m.b.x >= 2)
    m.b.secondTerm[2].cons = Constraint(expr=m.b.x >= 3)
    m.b.disjunctionList = Disjunction(Any)
    m.b.obj = Objective(expr=m.b.x)
    for i in range(1, 3):
        m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]
        TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])
        if i == 1:
            check_relaxation_block(self, m.b, '_pyomo_gdp_%s_reformulation' % transformation, 2)
        if i == 2:
            check_relaxation_block(self, m.b, '_pyomo_gdp_%s_reformulation_4' % transformation, 2)