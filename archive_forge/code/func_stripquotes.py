from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def stripquotes(s):
    """Replace explicit quotes in a string.

    >>> from sympy.physics.quantum.qasm import stripquotes
    >>> stripquotes("'S'") == 'S'
    True
    >>> stripquotes('"S"') == 'S'
    True
    >>> stripquotes('S') == 'S'
    True
    """
    s = s.replace('"', '')
    s = s.replace("'", '')
    return s