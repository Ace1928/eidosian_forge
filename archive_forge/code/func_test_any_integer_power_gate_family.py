import pytest
import sympy
import cirq
from cirq.ops.gateset_test import CustomX, CustomXPowGate
def test_any_integer_power_gate_family():
    with pytest.raises(ValueError, match='subclass of `cirq.EigenGate`'):
        cirq.AnyIntegerPowerGateFamily(gate=cirq.testing.SingleQubitGate)
    with pytest.raises(ValueError, match='subclass of `cirq.EigenGate`'):
        cirq.AnyIntegerPowerGateFamily(gate=CustomXPowGate())
    eq = cirq.testing.EqualsTester()
    gate_family = cirq.AnyIntegerPowerGateFamily(CustomXPowGate)
    eq.add_equality_group(gate_family)
    eq.add_equality_group(cirq.AnyIntegerPowerGateFamily(cirq.EigenGate))
    cirq.testing.assert_equivalent_repr(gate_family)
    assert CustomX in gate_family
    assert CustomX ** 2 in gate_family
    assert CustomX ** 1.5 not in gate_family
    assert CustomX ** sympy.Symbol('theta') not in gate_family
    assert 'CustomXPowGate' in gate_family.name
    assert '`g.exponent` is an integer' in gate_family.description