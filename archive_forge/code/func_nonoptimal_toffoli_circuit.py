from typing import TYPE_CHECKING
from cirq import ops, circuits
def nonoptimal_toffoli_circuit(q0: 'cirq.Qid', q1: 'cirq.Qid', q2: 'cirq.Qid') -> circuits.Circuit:
    ret = circuits.Circuit(ops.Y(q2) ** 0.5, ops.X(q2), ops.CNOT(q1, q2), ops.Z(q2) ** (-0.25), ops.CNOT(q1, q2), ops.CNOT(q2, q1), ops.CNOT(q1, q2), ops.CNOT(q0, q1), ops.CNOT(q1, q2), ops.CNOT(q2, q1), ops.CNOT(q1, q2), ops.Z(q2) ** 0.25, ops.CNOT(q1, q2), ops.Z(q2) ** (-0.25), ops.CNOT(q1, q2), ops.CNOT(q2, q1), ops.CNOT(q1, q2), ops.CNOT(q0, q1), ops.CNOT(q1, q2), ops.CNOT(q2, q1), ops.CNOT(q1, q2), ops.Z(q2) ** 0.25, ops.Z(q1) ** 0.25, ops.CNOT(q0, q1), ops.Z(q0) ** 0.25, ops.Z(q1) ** (-0.25), ops.CNOT(q0, q1), ops.Y(q2) ** 0.5, ops.X(q2))
    return ret