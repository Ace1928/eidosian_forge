from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_largest_collect_unused_collects_all_for_empty():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q12 = cirq.GridQubit(1, 2)
    qubits = [q00, q01, q02, q12]
    start = q01
    search = greedy._PickLargestArea(_create_device(qubits), start)
    assert search._collect_unused(start, set()) == set(qubits)
    assert search._collect_unused(start, {start}) == set(qubits)