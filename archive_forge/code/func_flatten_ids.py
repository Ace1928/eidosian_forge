from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def flatten_ids(ids):
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids, key=default_sort_key)
    ids = reduce(collapse, ids, [])
    ids.sort(key=default_sort_key)
    return ids