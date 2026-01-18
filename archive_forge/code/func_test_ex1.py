from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_ex1():
    if not mpl:
        skip('matplotlib not installed')
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot
    c = CircuitPlot(CNOT(1, 0) * H(1), 2, labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']