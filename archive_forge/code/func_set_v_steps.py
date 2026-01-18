from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def set_v_steps(self, v_steps):
    if v_steps is None:
        self._v_steps = None
        return
    if isinstance(v_steps, int):
        v_steps = Integer(v_steps)
    elif not isinstance(v_steps, Integer):
        raise ValueError('v_steps must be an int or SymPy Integer.')
    if v_steps <= S.Zero:
        raise ValueError('v_steps must be positive.')
    self._v_steps = v_steps