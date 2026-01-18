from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_labeller():
    """Test the labeller utility"""
    assert labeller(2) == ['q_1', 'q_0']
    assert labeller(3, 'j') == ['j_2', 'j_1', 'j_0']