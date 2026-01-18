from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_largest_sequence_search_does_not_use_used():
    q00 = cirq.GridQubit(0, 0)
    q10 = cirq.GridQubit(1, 0)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)
    qubits = [q00, q10, q20, q21]
    search = greedy._PickLargestArea(_create_device(qubits), q10)
    assert search._choose_next_qubit(q10, {q10, q20}) == q00