from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
def test_find_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    x1 = X(1)
    y1 = Y(1)
    i0 = Symbol('i0')
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)
    h_i0 = H(i0)
    circuit = (x, y, z)
    assert find_subcircuit(circuit, (x,)) == 0
    assert find_subcircuit(circuit, (x1,)) == -1
    assert find_subcircuit(circuit, (y,)) == 1
    assert find_subcircuit(circuit, (h,)) == -1
    assert find_subcircuit(circuit, Mul(x, h)) == -1
    assert find_subcircuit(circuit, Mul(x, y, z)) == 0
    assert find_subcircuit(circuit, Mul(y, z)) == 1
    assert find_subcircuit(Mul(*circuit), (x, y, z, h)) == -1
    assert find_subcircuit(Mul(*circuit), (z, y, x)) == -1
    assert find_subcircuit(circuit, (x,), start=2, end=1) == -1
    circuit = (x, y, x, y, z)
    assert find_subcircuit(Mul(*circuit), Mul(x, y, z)) == 2
    assert find_subcircuit(circuit, (x,), start=1) == 2
    assert find_subcircuit(circuit, (x, y), start=1, end=2) == -1
    assert find_subcircuit(Mul(*circuit), (x, y), start=1, end=3) == -1
    assert find_subcircuit(circuit, (x, y), start=1, end=4) == 2
    assert find_subcircuit(circuit, (x, y), start=2, end=4) == 2
    circuit = (x, y, z, x1, x, y, z, h, x, y, x1, x, y, z, h, y1, h)
    assert find_subcircuit(circuit, (x, y, z, h, y1)) == 11
    circuit = (x, y, x_i0, y_i0, z_i0, z)
    assert find_subcircuit(circuit, (x_i0, y_i0, z_i0)) == 2
    circuit = (x_i0, y_i0, z_i0, x_i0, y_i0, h_i0)
    subcircuit = (x_i0, y_i0, z_i0)
    result = find_subcircuit(circuit, subcircuit)
    assert result == 0