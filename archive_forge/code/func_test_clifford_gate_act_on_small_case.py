import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_act_on_small_case():
    qubits = cirq.LineQubit.range(5)
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=5), qubits=qubits, prng=np.random.RandomState())
    expected_args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=5), qubits=qubits, prng=np.random.RandomState())
    cirq.act_on(cirq.H, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.H, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau
    cirq.act_on(cirq.CNOT, expected_args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    assert args.tableau == expected_args.tableau
    cirq.act_on(cirq.H, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.H, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau
    cirq.act_on(cirq.S, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.S, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau
    cirq.act_on(cirq.X, expected_args, qubits=[qubits[2]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.X, args, qubits=[qubits[2]], allow_decompose=False)
    assert args.tableau == expected_args.tableau