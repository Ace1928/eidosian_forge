from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_is_reducible():
    nqubits = 2
    x, y, z, h = create_gate_sequence()
    circuit = (x, y, y)
    assert is_reducible(circuit, nqubits, 1, 3) is True
    circuit = (x, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is False
    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 0, 4) is True
    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is True
    circuit = (x, y, z, y, y)
    assert is_reducible(circuit, nqubits, 1, 5) is True