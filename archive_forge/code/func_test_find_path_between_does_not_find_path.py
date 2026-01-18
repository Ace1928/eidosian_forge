from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_find_path_between_does_not_find_path():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q22 = cirq.GridQubit(2, 2)
    q12 = cirq.GridQubit(1, 2)
    qubits = [q00, q01]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None
    qubits = [q00, q01, q10]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q01, {q00, q01}) is None
    qubits = [q00, q01, q02, q10, q20, q22, q12]
    path_1 = [q00, q01, q02]
    start = q00
    search = greedy.GreedySequenceSearch(_create_device(qubits), start)
    assert search._find_path_between(q00, q02, set(path_1)) is None