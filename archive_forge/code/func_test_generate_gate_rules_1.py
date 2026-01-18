from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_generate_gate_rules_1():
    x, y, z, h = create_gate_sequence()
    ph = PhaseGate(0)
    cgate_t = CGate(0, TGate(1))
    assert generate_gate_rules((x,)) == {((x,), ())}
    gate_rules = {((x, x), ()), ((x,), (x,))}
    assert generate_gate_rules((x, x)) == gate_rules
    gate_rules = {((x, y, x), ()), ((y, x, x), ()), ((x, x, y), ()), ((y, x), (x,)), ((x, y), (x,)), ((y,), (x, x))}
    assert generate_gate_rules((x, y, x)) == gate_rules
    gate_rules = {((x, y, z), ()), ((y, z, x), ()), ((z, x, y), ()), ((), (x, z, y)), ((), (y, x, z)), ((), (z, y, x)), ((x,), (z, y)), ((y, z), (x,)), ((y,), (x, z)), ((z, x), (y,)), ((z,), (y, x)), ((x, y), (z,))}
    actual = generate_gate_rules((x, y, z))
    assert actual == gate_rules
    gate_rules = {((), (h, z, y, x)), ((), (x, h, z, y)), ((), (y, x, h, z)), ((), (z, y, x, h)), ((h,), (z, y, x)), ((x,), (h, z, y)), ((y,), (x, h, z)), ((z,), (y, x, h)), ((h, x), (z, y)), ((x, y), (h, z)), ((y, z), (x, h)), ((z, h), (y, x)), ((h, x, y), (z,)), ((x, y, z), (h,)), ((y, z, h), (x,)), ((z, h, x), (y,)), ((h, x, y, z), ()), ((x, y, z, h), ()), ((y, z, h, x), ()), ((z, h, x, y), ())}
    actual = generate_gate_rules((x, y, z, h))
    assert actual == gate_rules
    gate_rules = {((), (cgate_t ** (-1), ph ** (-1), x)), ((), (ph ** (-1), x, cgate_t ** (-1))), ((), (x, cgate_t ** (-1), ph ** (-1))), ((cgate_t,), (ph ** (-1), x)), ((ph,), (x, cgate_t ** (-1))), ((x,), (cgate_t ** (-1), ph ** (-1))), ((cgate_t, x), (ph ** (-1),)), ((ph, cgate_t), (x,)), ((x, ph), (cgate_t ** (-1),)), ((cgate_t, x, ph), ()), ((ph, cgate_t, x), ()), ((x, ph, cgate_t), ())}
    actual = generate_gate_rules((x, ph, cgate_t))
    assert actual == gate_rules
    gate_rules = {(Integer(1), cgate_t ** (-1) * ph ** (-1) * x), (Integer(1), ph ** (-1) * x * cgate_t ** (-1)), (Integer(1), x * cgate_t ** (-1) * ph ** (-1)), (cgate_t, ph ** (-1) * x), (ph, x * cgate_t ** (-1)), (x, cgate_t ** (-1) * ph ** (-1)), (cgate_t * x, ph ** (-1)), (ph * cgate_t, x), (x * ph, cgate_t ** (-1)), (cgate_t * x * ph, Integer(1)), (ph * cgate_t * x, Integer(1)), (x * ph * cgate_t, Integer(1))}
    actual = generate_gate_rules((x, ph, cgate_t), return_as_muls=True)
    assert actual == gate_rules