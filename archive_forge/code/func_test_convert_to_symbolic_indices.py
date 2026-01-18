from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
from sympy.testing.pytest import slow
def test_convert_to_symbolic_indices():
    x, y, z, h = create_gate_sequence()
    i0 = Symbol('i0')
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x,))
    assert actual == (X(i0),)
    assert act_map == exp_map
    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h))
    assert actual == expected
    assert exp_map == act_map
    x1, y1, z1, h1 = create_gate_sequence(1)
    i1 = Symbol('i1')
    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map
    expected = (X(i0), Y(i0), Z(i0), H(i0), X(i1), Y(i1), Z(i1), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h, x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x1, y1, z1, h1, x, y, z, h))
    assert actual == expected
    assert act_map == exp_map
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x, x1, y, y1, z, z1, h, h1))
    assert actual == expected
    assert act_map == exp_map
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, x, y1, y, z1, z, h1, h))
    assert actual == expected
    assert act_map == exp_map
    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1), CNOT(i1, i0), CNOT(i0, i1), CGate(i1, Z(i0)), CGate(i0, Z(i1)))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    args = (x, x1, y, y1, z, z1, h, h1, cnot_10, cnot_01, cgate_z_10, cgate_z_01)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    args = (x1, x, y1, y, z1, z, h1, h, cnot_10, cnot_01, cgate_z_10, cgate_z_01)
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1), CNOT(i0, i1), CNOT(i1, i0), CGate(i0, Z(i1)), CGate(i1, Z(i0)))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    args = (cnot_10, h, cgate_z_01, h)
    expected = (CNOT(i0, i1), H(i1), CGate(i1, Z(i0)), H(i1))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    args = (cnot_01, h1, cgate_z_10, h1)
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    args = (cnot_10, h1, cgate_z_01, h1)
    expected = (CNOT(i0, i1), H(i0), CGate(i1, Z(i0)), H(i0))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    i2 = Symbol('i2')
    ccgate_z = CGate(0, CGate(1, Z(2)))
    ccgate_x = CGate(1, CGate(2, X(0)))
    args = (ccgate_z, ccgate_x)
    expected = (CGate(i0, CGate(i1, Z(i2))), CGate(i1, CGate(i2, X(i0))))
    exp_map = {i0: Integer(0), i1: Integer(1), i2: Integer(2)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map
    ndx_map = {i0: Integer(0)}
    index_gen = numbered_symbols(prefix='i', start=1)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args, qubit_map=ndx_map, start=i0, gen=index_gen)
    assert actual == expected
    assert act_map == exp_map
    i3 = Symbol('i3')
    cgate_x0_c321 = CGate((3, 2, 1), X(0))
    exp_map = {i0: Integer(3), i1: Integer(2), i2: Integer(1), i3: Integer(0)}
    expected = (CGate((i0, i1, i2), X(i3)),)
    args = (cgate_x0_c321,)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map