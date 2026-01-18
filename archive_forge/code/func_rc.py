from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent
from .sparse import MutableSparseMatrix as SparseMatrix
def rc(d):
    r = -d if d < 0 else 0
    c = 0 if r else d
    return (r, c)