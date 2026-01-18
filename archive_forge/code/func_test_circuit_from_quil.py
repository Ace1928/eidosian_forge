import numpy as np
import pytest
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from cirq import Circuit, LineQubit
from cirq_rigetti.quil_input import (
from cirq.ops import (
def test_circuit_from_quil():
    q0, q1, q2 = LineQubit.range(3)
    cirq_circuit = Circuit([I(q0), I(q1), I(q2), X(q0), Y(q1), Z(q2), H(q0), S(q1), T(q2), Z(q0) ** (1 / 8), Z(q1) ** (1 / 8), Z(q2) ** (1 / 8), rx(np.pi / 2)(q0), ry(np.pi / 2)(q1), rz(np.pi / 2)(q2), CZ(q0, q1), CNOT(q1, q2), cphase(np.pi / 2)(q0, q1), cphase00(np.pi / 2)(q1, q2), cphase01(np.pi / 2)(q0, q1), cphase10(np.pi / 2)(q1, q2), ISWAP(q0, q1), pswap(np.pi / 2)(q1, q2), SWAP(q0, q1), xy(np.pi / 2)(q1, q2), CCNOT(q0, q1, q2), CSWAP(q0, q1, q2), MeasurementGate(1, key='ro[0]')(q0), MeasurementGate(1, key='ro[1]')(q1), MeasurementGate(1, key='ro[2]')(q2)])
    quil_circuit = circuit_from_quil(QUIL_PROGRAM)
    assert cirq_circuit == quil_circuit
    pyquil_circuit = Program(QUIL_PROGRAM)
    pyquil_unitary = program_unitary(pyquil_circuit[1:-3], n_qubits=3)
    cirq_circuit_swapped = Circuit(SWAP(q0, q2), cirq_circuit[:-1], SWAP(q0, q2))
    cirq_unitary = cirq_circuit_swapped.unitary()
    assert np.isclose(pyquil_unitary, cirq_unitary).all()