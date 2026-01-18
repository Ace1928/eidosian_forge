from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar
def coherent_state(n, alpha):
    """
    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.
    See https://en.wikipedia.org/wiki/Coherent_states

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    alpha :
        The eigen value of annihilation operator.
    """
    return exp(-Abs(alpha) ** 2 / 2) * alpha ** n / sqrt(factorial(n))