import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('input_gate, assert_implemented', [(cirq.X, True), (cirq.Y, True), (cirq.Z, True), (cirq.X ** 0.5, True), (cirq.Y ** 0.5, True), (cirq.Z ** 0.5, True), (cirq.X ** 3.5, True), (cirq.Y ** 3.5, True), (cirq.Z ** 3.5, True), (cirq.X ** 4, True), (cirq.Y ** 4, True), (cirq.Z ** 4, True), (cirq.H, True), (cirq.CX, True), (cirq.CZ, True), (cirq.H ** 4, True), (cirq.CX ** 4, True), (cirq.CZ ** 4, True), (cirq.X ** 0.25, False), (cirq.Y ** 0.25, False), (cirq.Z ** 0.25, False), (cirq.H ** 0.5, False), (cirq.CX ** 0.5, False), (cirq.CZ ** 0.5, False)])
def test_act_on_consistency(input_gate, assert_implemented):
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(input_gate, assert_implemented, assert_implemented)