from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_bfs_identity_search():
    assert bfs_identity_search([], 1) == set()
    x, y, z, h = create_gate_sequence()
    gate_list = [x]
    id_set = {GateIdentity(x, x)}
    assert bfs_identity_search(gate_list, 1, max_depth=2) == id_set
    gate_list = [x, y, z]
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(x, y, z)}
    assert bfs_identity_search(gate_list, 1) == id_set
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(x, y, z), GateIdentity(x, y, x, y), GateIdentity(x, z, x, z), GateIdentity(y, z, y, z)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set
    gate_list = [x, y, z, h]
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(h, h), GateIdentity(x, y, z), GateIdentity(x, y, x, y), GateIdentity(x, z, x, z), GateIdentity(x, h, z, h), GateIdentity(y, z, y, z), GateIdentity(y, h, y, h)}
    assert bfs_identity_search(gate_list, 1) == id_set
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(h, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=3, identity_only=True)
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(h, h), GateIdentity(x, y, z), GateIdentity(x, y, x, y), GateIdentity(x, z, x, z), GateIdentity(x, h, z, h), GateIdentity(y, z, y, z), GateIdentity(y, h, y, h), GateIdentity(x, y, h, x, h), GateIdentity(x, z, h, y, h), GateIdentity(y, z, h, z, h)}
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set
    id_set = {GateIdentity(x, x), GateIdentity(y, y), GateIdentity(z, z), GateIdentity(h, h), GateIdentity(x, h, z, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=4, identity_only=True)
    cnot = CNOT(1, 0)
    gate_list = [x, cnot]
    id_set = {GateIdentity(x, x), GateIdentity(cnot, cnot), GateIdentity(x, cnot, x, cnot)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set
    cgate_x = CGate((1,), x)
    gate_list = [x, cgate_x]
    id_set = {GateIdentity(x, x), GateIdentity(cgate_x, cgate_x), GateIdentity(x, cgate_x, x, cgate_x)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set
    cgate_z = CGate((0,), Z(1))
    gate_list = [cnot, cgate_z, h]
    id_set = {GateIdentity(h, h), GateIdentity(cgate_z, cgate_z), GateIdentity(cnot, cnot), GateIdentity(cnot, h, cgate_z, h)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set
    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [s, t]
    id_set = {GateIdentity(s, s, s, s)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set