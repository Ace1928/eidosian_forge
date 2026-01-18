from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_is_scalar_nonsparse_matrix():
    numqubits = 2
    id_only = False
    id_gate = (IdentityGate(1),)
    actual = is_scalar_nonsparse_matrix(id_gate, numqubits, id_only)
    assert actual is True
    x0 = X(0)
    xx_circuit = (x0, x0)
    actual = is_scalar_nonsparse_matrix(xx_circuit, numqubits, id_only)
    assert actual is True
    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    actual = is_scalar_nonsparse_matrix(xy_circuit, numqubits, id_only)
    assert actual is False
    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is True
    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True
    h = H(0)
    hh_circuit = (h, h)
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True
    id_only = True
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is False
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True