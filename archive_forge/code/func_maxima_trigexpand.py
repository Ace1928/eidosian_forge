import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
def maxima_trigexpand(expr):
    return expr.expand(trig=True)