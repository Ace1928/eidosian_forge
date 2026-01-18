from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def set_v_max(self, v_max):
    if v_max is None:
        self._v_max = None
        return
    try:
        self._v_max = sympify(v_max)
        float(self._v_max.evalf())
    except TypeError:
        raise ValueError('v_max could not be interpreted as a number.')