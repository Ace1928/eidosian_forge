from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_minimal_sequence_search_returns_none_for_single_node():
    q00 = cirq.GridQubit(0, 0)
    qubits = [q00]
    search = greedy._PickFewestNeighbors(_create_device(qubits), q00)
    assert search._choose_next_qubit(q00, {q00}) is None