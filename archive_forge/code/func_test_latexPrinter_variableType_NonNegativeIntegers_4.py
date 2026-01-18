import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
from pyomo.common.errors import InfeasibleConstraintException
def test_latexPrinter_variableType_NonNegativeIntegers_4(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 0))
    m.objective = pyo.Objective(expr=m.x)
    m.constraint_1 = pyo.Constraint(expr=m.x ** 2 <= 5.0)
    pstr = latex_printer(m)
    bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{s.t.} \n            & & x^{2} \\leq 5 & \\label{con:basicFormulation_constraint_1} \\\\ \n            & \\text{w.b.} \n            & &  0 \\leq x = 0  & \\qquad \\in \\mathds{Z}_{\\geq 0} \\label{con:basicFormulation_x_bound} \n        \\end{align} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)