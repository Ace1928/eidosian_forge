import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_throwTemplatizeError(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.x = pyo.Var(m.I, bounds=[-10, 10])
    m.c = pyo.Param(m.I, initialize=1.0, mutable=True)

    def ruleMaker_1(m):
        return sum((m.c[i] * m.x[i] for i in m.I))

    def ruleMaker_2(m, i):
        if i >= 2:
            return m.x[i] <= 1
        else:
            return pyo.Constraint.Skip
    m.objective = pyo.Objective(rule=ruleMaker_1)
    m.constraint_1 = pyo.Constraint(m.I, rule=ruleMaker_2)
    self.assertRaises(RuntimeError, latex_printer, **{'pyomo_component': m, 'throw_templatization_error': True})
    pstr = latex_printer(m)
    bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & \\sum_{ i \\in I  } c_{i} x_{i} & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & x[2] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[3] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[4] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & & x[5] \\leq 1 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & \\text{w.b.} \n            & & -10 \\leq x \\leq 10 & \\qquad \\in \\mathds{R} \\label{con:basicFormulation_x_bound} \n        \\end{align} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)