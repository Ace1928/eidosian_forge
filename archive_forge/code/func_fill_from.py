from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer
def fill_from(self, b):
    if b.v is not None:
        self.v = b.v
    if b.v_min is not None:
        self.v_min = b.v_min
    if b.v_max is not None:
        self.v_max = b.v_max
    if b.v_steps is not None:
        self.v_steps = b.v_steps