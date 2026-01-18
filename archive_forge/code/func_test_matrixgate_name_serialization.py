import re
import numpy as np
import pytest
import sympy
import cirq
def test_matrixgate_name_serialization():
    gate1 = cirq.MatrixGate(np.eye(2), name='test_name')
    gate_after_serialization1 = cirq.read_json(json_text=cirq.to_json(gate1))
    assert gate1._name == 'test_name'
    assert gate_after_serialization1._name == 'test_name'
    gate2 = cirq.MatrixGate(np.eye(2))
    gate_after_serialization2 = cirq.read_json(json_text=cirq.to_json(gate2))
    assert gate2._name is None
    assert gate_after_serialization2._name is None
    gate3 = cirq.MatrixGate(np.eye(2), name='')
    gate_after_serialization3 = cirq.read_json(json_text=cirq.to_json(gate3))
    assert gate3._name == ''
    assert gate_after_serialization3._name == ''