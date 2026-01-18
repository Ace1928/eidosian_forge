from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_get_or_search_calls_find_sequence_once():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    search = greedy.GreedySequenceSearch(_create_device([q00, q01]), q00)
    with mock.patch.object(search, '_find_sequence') as find_sequence:
        sequence = [q00, q01]
        find_sequence.return_value = sequence
        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()
        assert search.get_or_search() == sequence
        find_sequence.assert_called_once_with()