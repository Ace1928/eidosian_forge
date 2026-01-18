from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def nonblank(args):
    for line in args:
        line = trim(line)
        if line.isspace():
            continue
        yield line
    return