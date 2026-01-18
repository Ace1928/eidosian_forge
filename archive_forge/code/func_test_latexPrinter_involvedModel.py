import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def test_latexPrinter_involvedModel(self):
    m = generate_model()
    pstr = latex_printer(m)
    print(pstr)
    bstr = dedent('\n        \\begin{align} \n            & \\min \n            & & x + y + z & \\label{obj:basicFormulation_objective_1} \\\\ \n            & \\min \n            & &  \\left( x + y \\right)  \\sum_{ i \\in J  } w_{i} & \\label{obj:basicFormulation_objective_2} \\\\ \n            & \\max \n            & & x + y + z & \\label{obj:basicFormulation_objective_3} \\\\ \n            & \\text{s.t.} \n            & & x^{2} + y^{-2} - x y z + 1 = 2 & \\label{con:basicFormulation_constraint_1} \\\\ \n            &&&  \\left| \\frac{x}{z^{-2}} \\right|   \\left( x + y \\right)  \\leq 2 & \\label{con:basicFormulation_constraint_2} \\\\ \n            &&& \\sqrt { \\frac{x}{z^{-2}} } \\leq 2 & \\label{con:basicFormulation_constraint_3} \\\\ \n            &&& 1 \\leq x \\leq 2 & \\label{con:basicFormulation_constraint_4} \\\\ \n            &&& f_{\\text{exprIf}}(x \\leq 1,z,y) \\leq 1 & \\label{con:basicFormulation_constraint_5} \\\\ \n            &&& x + f\\_1(x,y) = 2 & \\label{con:basicFormulation_constraint_6} \\\\ \n            &&&  \\left( x + y \\right)  \\sum_{ i \\in I  } v_{i} + u_{i,j}^{2} \\leq 0 &  \\qquad \\forall j \\in I \\label{con:basicFormulation_constraint_7} \\\\ \n            &&& \\sum_{ i \\in K  } p_{i} = 1 & \\label{con:basicFormulation_constraint_8} \n        \\end{align} \n        ')
    self.assertEqual('\n' + pstr + '\n', bstr)