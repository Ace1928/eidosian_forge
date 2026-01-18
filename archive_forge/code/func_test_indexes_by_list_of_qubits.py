import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_indexes_by_list_of_qubits():
    q = cirq.LineQubit.range(4)
    moment = cirq.Moment([cirq.Z(q[0]), cirq.CNOT(q[1], q[2])])
    assert moment[[q[0]]] == cirq.Moment([cirq.Z(q[0])])
    assert moment[[q[1]]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[2]]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[3]]] == cirq.Moment([])
    assert moment[q[0:2]] == moment
    assert moment[q[1:3]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[q[2:4]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[0], q[3]]] == cirq.Moment([cirq.Z(q[0])])
    assert moment[q] == moment