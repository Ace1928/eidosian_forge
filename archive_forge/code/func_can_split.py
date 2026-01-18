import sympy
from sympy.parsing.sympy_parser import (
from sympy.testing.pytest import raises
def can_split(symbol):
    if symbol not in ('unsplittable', 'names'):
        return _token_splittable(symbol)
    return False