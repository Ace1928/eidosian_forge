import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
def test_decompose_two_qubit_interaction_into_four_fsim_gates_validate():
    iswap = cirq.FSimGate(theta=np.pi / 2, phi=0)
    with pytest.raises(ValueError, match='fsim_gate.theta'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=cirq.FSimGate(theta=np.pi / 10, phi=0))
    with pytest.raises(ValueError, match='fsim_gate.phi'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 3))
    with pytest.raises(ValueError, match='pair of qubits'):
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=iswap, qubits=cirq.LineQubit.range(3))
    with pytest.raises(ValueError, match='parameterized'):
        fsim = cirq.FSimGate(theta=np.pi / 2, phi=sympy.Symbol('x'))
        cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=fsim)