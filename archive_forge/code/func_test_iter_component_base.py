from itertools import zip_longest
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.kernel as pmo
from pyomo.util.components import iter_component, rename_components
def test_iter_component_base(self):
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2, 3], initialize=0)
    model.z = pyo.Var(initialize=0)

    def con_rule(m, i):
        return m.x[i] + m.z == i
    model.con = pyo.Constraint([1, 2, 3], rule=con_rule)
    model.zcon = pyo.Constraint(expr=model.z >= model.x[2])
    self.assertSameComponents(list(iter_component(model.x)), list(model.x.values()))
    self.assertSameComponents(list(iter_component(model.z)), [model.z[None]])
    self.assertSameComponents(list(iter_component(model.con)), list(model.con.values()))
    self.assertSameComponents(list(iter_component(model.zcon)), [model.zcon[None]])