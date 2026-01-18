from sympy.assumptions import ask, Q
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
def is_extended_real(obj, assumptions=None):
    if assumptions is None:
        return obj.is_extended_real
    return ask(Q.extended_real(obj), assumptions)