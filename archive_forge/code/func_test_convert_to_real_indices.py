from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
def test_convert_to_real_indices():
    i0 = Symbol('i0')
    i1 = Symbol('i1')
    x, y, z, h = create_gate_sequence()
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)
    qubit_map = {i0: 0}
    args = (z_i0, y_i0, x_i0)
    expected = (z, y, x)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))
    cnot_i1_i0 = CNOT(i1, i0)
    cnot_i0_i1 = CNOT(i0, i1)
    cgate_z_i1_i0 = CGate(i1, Z(i0))
    qubit_map = {i0: 0, i1: 1}
    args = (cnot_i1_i0,)
    expected = (cnot_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    args = (cnot_i0_i1,)
    expected = (cnot_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    qubit_map = {i0: 1, i1: 0}
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected
    i2 = Symbol('i2')
    ccgate_z = CGate(i0, CGate(i1, Z(i2)))
    ccgate_x = CGate(i1, CGate(i2, X(i0)))
    qubit_map = {i0: 0, i1: 1, i2: 2}
    args = (ccgate_z, ccgate_x)
    expected = (CGate(0, CGate(1, Z(2))), CGate(1, CGate(2, X(0))))
    actual = convert_to_real_indices(Mul(*args), qubit_map)
    assert actual == expected
    qubit_map = {i0: 1, i2: 0, i1: 2}
    args = (ccgate_x, ccgate_z)
    expected = (CGate(2, CGate(0, X(1))), CGate(1, CGate(2, Z(0))))
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected