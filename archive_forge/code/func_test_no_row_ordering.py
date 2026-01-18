import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_no_row_ordering(self):
    model = ConcreteModel()
    model.a = Var()
    components = {}
    components['obj'] = Objective(expr=model.a)
    components['con1'] = Constraint(expr=model.a >= 0)
    components['con2'] = Constraint(expr=model.a <= 1)
    components['con3'] = Constraint(expr=(0, model.a, 1))
    components['con4'] = Constraint([1, 2], rule=lambda m, i: model.a == i)
    for key in components:
        model.add_component(key, components[key])
    self._check_baseline(model, file_determinism=2)