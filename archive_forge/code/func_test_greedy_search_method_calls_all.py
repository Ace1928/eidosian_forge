from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_greedy_search_method_calls_all():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    qubits = [q00, q01]
    length = 2
    method = greedy.GreedySequenceSearchStrategy()
    assert len(method.place_line(_create_device(qubits), length)) == 2