import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_no_column_ordering_linear(self):
    model = ConcreteModel()
    model.a = Var()
    model.b = Var()
    model.c = Var()
    terms = [model.a, model.b, model.c]
    model.obj = Objective(expr=self._gen_expression(terms))
    model.con = Constraint(expr=self._gen_expression(terms) <= 1)
    self._check_baseline(model)