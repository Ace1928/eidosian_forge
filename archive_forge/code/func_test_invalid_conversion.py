import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_invalid_conversion(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    with self.assertRaisesRegex(TypeError, 'argument must be a string or a(.*) number'):
        float(m.Y1)
    with self.assertRaisesRegex(TypeError, 'argument must be a string(?:, a bytes-like object)? or a(.*) number'):
        int(m.Y1)