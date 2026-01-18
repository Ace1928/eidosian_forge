from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_create1():
    Qgate = CreateOneQubitGate('Q')
    assert str(Qgate(0)) == 'Q(0)'