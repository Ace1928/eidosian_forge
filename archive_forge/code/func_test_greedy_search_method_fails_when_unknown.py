from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_greedy_search_method_fails_when_unknown():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    qubits = [q00, q01]
    length = 2
    method = greedy.GreedySequenceSearchStrategy('fail')
    with pytest.raises(ValueError):
        method.place_line(_create_device(qubits), length)