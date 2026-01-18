from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def set_v(self, v):
    if v is None:
        self._v = None
        return
    if not isinstance(v, Symbol):
        raise ValueError('v must be a SymPy Symbol.')
    self._v = v