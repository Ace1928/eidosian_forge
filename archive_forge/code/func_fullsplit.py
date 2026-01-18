from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def fullsplit(line):
    words = line.split()
    rest = ' '.join(words[1:])
    return (fixcommand(words[0]), [s.strip() for s in rest.split(',')])