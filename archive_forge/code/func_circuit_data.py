from typing import cast, Tuple, List
import cirq
import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
from cirq_rigetti import circuit_transformers
@pytest.fixture
def circuit_data() -> Tuple[cirq.Circuit, List[cirq.LineQubit], cirq.Linspace]:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.X(qubits[0]) ** sympy.Symbol('t'))
    circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))
    param_sweep = cirq.Linspace('t', start=0, stop=2, length=5)
    return (circuit, qubits, param_sweep)