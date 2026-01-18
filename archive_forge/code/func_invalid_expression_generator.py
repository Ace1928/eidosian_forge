import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def invalid_expression_generator():
    yield (lambda: m.Y1 + m.Y2)
    yield (lambda: m.Y1 - m.Y2)
    yield (lambda: m.Y1 * m.Y2)
    yield (lambda: m.Y1 / m.Y2)
    yield (lambda: m.Y1 ** m.Y2)
    yield (lambda: m.Y1.land(0))
    yield (lambda: m.Y1.lor(0))
    yield (lambda: m.Y1.xor(0))
    yield (lambda: m.Y1.equivalent_to(0))
    yield (lambda: m.Y1.implies(0))
    yield (lambda: 0 + m.Y2)
    yield (lambda: 0 - m.Y2)
    yield (lambda: 0 * m.Y2)
    yield (lambda: 0 / m.Y2)
    yield (lambda: 0 ** m.Y2)
    yield (lambda: 0 & m.Y2)
    yield (lambda: 0 | m.Y2)
    yield (lambda: 0 ^ m.Y2)
    yield (lambda: m.Y3 + 2)
    yield (lambda: m.Y3 - 2)
    yield (lambda: m.Y3 * 2)
    yield (lambda: m.Y3 / 2)
    yield (lambda: m.Y3 ** 2)
    yield (lambda: m.Y3 & 2)
    yield (lambda: m.Y3 | 2)
    yield (lambda: m.Y3 ^ 2)
    yield iadd
    yield isub
    yield imul
    yield idiv
    yield ipow
    yield iand
    yield ior
    yield ixor