from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_noise_composition():
    a, b, c = cirq.LineQubit.range(3)
    noise_z = cirq.ConstantQubitNoiseModel(cirq.Z)
    noise_inv_s = cirq.ConstantQubitNoiseModel(cirq.S ** (-1))
    base_moments = [cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)]), cirq.Moment([cirq.H(c)])]
    circuit_z = cirq.Circuit(noise_z.noisy_moments(base_moments, [a, b, c]))
    circuit_s = cirq.Circuit(noise_inv_s.noisy_moments(base_moments, [a, b, c]))
    actual_zs = cirq.Circuit(noise_inv_s.noisy_moments(circuit_z.moments, [a, b, c]))
    actual_sz = cirq.Circuit(noise_z.noisy_moments(circuit_s.moments, [a, b, c]))
    expected_circuit = cirq.Circuit(cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]), cirq.Moment([cirq.Y(b)]), cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]), cirq.Moment([cirq.H(c)]), cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]))
    actual_zs = cirq.merge_single_qubit_gates_to_phased_x_and_z(actual_zs)
    actual_sz = cirq.merge_single_qubit_gates_to_phased_x_and_z(actual_sz)
    expected_circuit = cirq.merge_single_qubit_gates_to_phased_x_and_z(expected_circuit)
    assert_equivalent_op_tree(actual_zs, actual_sz)
    assert_equivalent_op_tree(actual_zs, expected_circuit)