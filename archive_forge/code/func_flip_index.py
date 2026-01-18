from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def flip_index(i, n):
    """Reorder qubit indices from largest to smallest.

    >>> from sympy.physics.quantum.qasm import flip_index
    >>> flip_index(0, 2)
    1
    >>> flip_index(1, 2)
    0
    """
    return n - i - 1