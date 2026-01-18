import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_unary(self):
    m = generate_model()
    pstr = latex_printer(m.constraint_2)
    bstr = dedent('\n        \\begin{equation} \n              \\left| \\frac{x}{z^{-2}} \\right|   \\left( x + y \\right)  \\leq 2 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    pstr = latex_printer(pyo.Constraint(expr=pyo.sin(m.x) == 1))
    bstr = dedent('\n        \\begin{equation} \n             \\sin \\left( x \\right)  = 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    pstr = latex_printer(pyo.Constraint(expr=pyo.log10(m.x) == 1))
    bstr = dedent('\n        \\begin{equation} \n             \\log_{10} \\left( x \\right)  = 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)
    pstr = latex_printer(pyo.Constraint(expr=pyo.sqrt(m.x) == 1))
    bstr = dedent('\n        \\begin{equation} \n             \\sqrt { x } = 1 \n        \\end{equation} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)