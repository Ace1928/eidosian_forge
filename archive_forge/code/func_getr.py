import math
import random
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.numbers import igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction
from sympy.utilities.iterables import variations
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qft import QFT
from sympy.physics.quantum.qexpr import QuantumError
def getr(x, y, N):
    fraction = continued_fraction(x, y)
    total = ratioize(fraction, N)
    return total