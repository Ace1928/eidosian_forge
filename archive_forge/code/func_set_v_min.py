from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def set_v_min(self, v_min):
    if v_min is None:
        self._v_min = None
        return
    try:
        self._v_min = sympify(v_min)
        float(self._v_min.evalf())
    except TypeError:
        raise ValueError('v_min could not be interpreted as a number.')