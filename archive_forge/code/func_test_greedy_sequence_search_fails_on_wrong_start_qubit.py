from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_greedy_sequence_search_fails_on_wrong_start_qubit():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    with pytest.raises(ValueError):
        greedy.GreedySequenceSearch(_create_device([q00]), q01)