import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def test_binary_and(self):
    m = ConcreteModel()
    m.Y1 = BooleanVar()
    m.Y2 = BooleanVar()
    op_static = land(m.Y1, m.Y2)
    op_class = m.Y1.land(m.Y2)
    op_operator = m.Y1 & m.Y2
    for truth_combination in _generate_possible_truth_inputs(2):
        m.Y1.value, m.Y2.value = (truth_combination[0], truth_combination[1])
        correct_value = all(truth_combination)
        self.assertEqual(value(op_static), correct_value)
        self.assertEqual(value(op_class), correct_value)
        self.assertEqual(value(op_operator), correct_value)