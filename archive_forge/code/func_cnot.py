from math import prod
from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz
def cnot(self, a1, a2):
    self.circuit.append(CNOT(*self.indices([a1, a2])))