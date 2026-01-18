from typing import cast, Tuple
import cirq
import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
@pytest.fixture
def circuit_with_separate_readout_keys() -> Tuple[cirq.Circuit, cirq.Linspace]:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.X(qubits[0]) ** sympy.Symbol('t'))
    circuit.append(cirq.measure(qubits[0], key='m0'))
    circuit.append(cirq.measure(qubits[1], key='m1'))
    param_sweep = cirq.Linspace('t', start=0, stop=2, length=5)
    return (circuit, param_sweep)