import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_preserving_structure():
    a, b = cirq.LineQubit.range(2)
    fc1 = cirq.FrozenCircuit(cirq.SWAP(a, b), cirq.FSimGate(0.1, 0.2).on(a, b))
    cop1_1 = cirq.CircuitOperation(fc1).with_tags('test_tag')
    cop1_2 = cirq.CircuitOperation(fc1).with_qubit_mapping({a: b, b: a})
    fc2 = cirq.FrozenCircuit(cirq.X(a), cop1_1, cop1_2)
    cop2 = cirq.CircuitOperation(fc2)
    circuit = cirq.Circuit(cop2, cirq.measure(a, b, key='m'))
    actual = cirq.Circuit(cirq.decompose(circuit, preserve_structure=True))
    fc1_decomp = cirq.FrozenCircuit(cirq.decompose(fc1))
    expected = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(a), cirq.CircuitOperation(fc1_decomp).with_tags('test_tag'), cirq.CircuitOperation(fc1_decomp).with_qubit_mapping({a: b, b: a}))), cirq.measure(a, b, key='m'))
    assert actual == expected