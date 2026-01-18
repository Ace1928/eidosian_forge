from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_greedy_search_method_returns_empty_when_empty():
    device = _create_device([])
    length = 0
    method = greedy.GreedySequenceSearchStrategy()
    assert method.place_line(device, length) == GridQubitLineTuple()