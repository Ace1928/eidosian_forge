from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def cx(self, a1, a2):
    fi, fj = self.indices([a1, a2])
    self.circuit.append(CGate(fi, X(fj)))