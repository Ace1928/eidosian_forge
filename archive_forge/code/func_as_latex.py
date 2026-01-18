from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from types import FunctionType
def as_latex(self):
    from .latex import latex
    return latex(self)