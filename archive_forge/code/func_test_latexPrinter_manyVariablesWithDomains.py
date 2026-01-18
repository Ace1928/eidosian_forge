import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_manyVariablesWithDomains(self):
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var(domain=Integers, bounds=(-10, 10))
    m.y = pyo.Var(domain=Binary, bounds=(-10, 10))
    m.z = pyo.Var(domain=PositiveReals, bounds=(-10, 10))
    m.u = pyo.Var(domain=NonNegativeIntegers, bounds=(-10, 10))
    m.v = pyo.Var(domain=NegativeReals, bounds=(-10, 10))
    m.w = pyo.Var(domain=PercentFraction, bounds=(-10, 10))
    m.objective = pyo.Objective(expr=m.x + m.y + m.z + m.u + m.v + m.w)
    pstr = latex_printer(m)
    bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z + u + v + w & \\label{obj:basicFormulation_objective} \\\\ \n            & \\text{w.b.} \n            & & -10 \\leq x \\leq 10 & \\qquad \\in \\mathds{Z} \\label{con:basicFormulation_x_bound} \\\\ \n            &&& y & \\qquad \\in \\left\\{ 0 , 1 \\right \\} \\label{con:basicFormulation_y_bound} \\\\ \n            &&&  0 < z \\leq 10 & \\qquad \\in \\mathds{R}_{> 0} \\label{con:basicFormulation_z_bound} \\\\ \n            &&&  0 \\leq u \\leq 10 & \\qquad \\in \\mathds{Z}_{\\geq 0} \\label{con:basicFormulation_u_bound} \\\\ \n            &&& -10 \\leq v < 0  & \\qquad \\in \\mathds{R}_{< 0} \\label{con:basicFormulation_v_bound} \\\\ \n            &&&  0 \\leq w \\leq 1  & \\qquad \\in \\mathds{R} \\label{con:basicFormulation_w_bound} \n        \\end{align} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)