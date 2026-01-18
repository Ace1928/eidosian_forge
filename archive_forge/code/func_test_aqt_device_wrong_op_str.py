from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
def test_aqt_device_wrong_op_str():
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit.range(2)
    circuit.append(cirq.CNOT(q0, q1) ** 1.0)
    for op in circuit.all_operations():
        with pytest.raises(ValueError):
            _result = get_op_string(op)