import pytest
import sympy
import numpy as np
import cirq
import cirq_pasqal
from cirq_pasqal import PasqalDevice, PasqalVirtualDevice
from cirq_pasqal import TwoDQubit, ThreeDQubit
def test_is_pasqal_device_op():
    d = generic_device(2)
    with pytest.raises(ValueError, match='Got unknown operation'):
        d.is_pasqal_device_op(cirq.NamedQubit('q0'))
    op = cirq.ops.CZ.on(*d.qubit_list())
    bad_op = cirq.ops.CNotPowGate(exponent=0.5)
    assert d.is_pasqal_device_op(op)
    assert d.is_pasqal_device_op(cirq.ops.X(cirq.NamedQubit('q0')))
    assert not d.is_pasqal_device_op(cirq.ops.CCX(cirq.NamedQubit('q0'), cirq.NamedQubit('q1'), cirq.NamedQubit('q2')) ** 0.2)
    assert not d.is_pasqal_device_op(bad_op(cirq.NamedQubit('q0'), cirq.NamedQubit('q1')))
    for op1 in [cirq.CNotPowGate(exponent=1.0), cirq.CNotPowGate(exponent=1.0, global_shift=-0.5)]:
        assert d.is_pasqal_device_op(op1(cirq.NamedQubit('q0'), cirq.NamedQubit('q1')))
    op2 = (cirq.ops.H ** sympy.Symbol('exp')).on(d.qubit_list()[0])
    assert not d.is_pasqal_device_op(op2)
    d2 = square_virtual_device(control_r=1.1, num_qubits=3)
    assert d.is_pasqal_device_op(cirq.ops.X(TwoDQubit(0, 0)))
    assert not d2.is_pasqal_device_op(op1(TwoDQubit(0, 0), TwoDQubit(0, 1)))