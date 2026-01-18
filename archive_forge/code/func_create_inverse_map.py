from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def create_inverse_map(symb_to_real_map):
    rev_items = lambda item: (item[1], item[0])
    return dict(map(rev_items, symb_to_real_map.items()))