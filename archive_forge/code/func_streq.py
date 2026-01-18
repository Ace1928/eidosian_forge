from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, oo, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.physics.optics import (BeamParameter, CurvedMirror,
def streq(a, b):
    return str(a) == str(b)