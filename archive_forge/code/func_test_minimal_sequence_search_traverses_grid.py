from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_minimal_sequence_search_traverses_grid():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q11 = cirq.GridQubit(1, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)
    q04 = cirq.GridQubit(0, 4)
    q14 = cirq.GridQubit(1, 4)
    q24 = cirq.GridQubit(2, 4)
    q05 = cirq.GridQubit(0, 5)
    qubits = [q00, q01, q11, q02, q03, q04, q05, q14, q24]
    device = _create_device(qubits)
    search = greedy._PickFewestNeighbors(device, q02)
    assert search._choose_next_qubit(q02, {q02}) == q03
    assert search._choose_next_qubit(q03, {q02, q03}) == q04
    assert search._choose_next_qubit(q04, {q02, q03, q04}) == q14
    assert search._choose_next_qubit(q14, {q02, q03, q04, q14}) == q24
    assert search._choose_next_qubit(q24, {q02, q03, q04, q14, q24}) is None
    assert search._choose_next_qubit(q24, {q24}) == q14
    assert search._choose_next_qubit(q14, {q24, q14}) == q04
    assert search._choose_next_qubit(q04, {q24, q14, q04}) == q03
    assert search._choose_next_qubit(q03, {q24, q14, q04, q03}) == q02
    assert search._choose_next_qubit(q02, {q24, q14, q04, q03, q02}) == q01
    assert search._choose_next_qubit(q01, {q24, q14, q04, q03, q02, q01}) in [q00, q11]
    assert search._choose_next_qubit(q00, {q24, q14, q04, q03, q02, q01, q00}) is None
    assert search._choose_next_qubit(q11, {q24, q14, q04, q03, q02, q01, q11}) is None
    qubits = [q00, q01, q02, q03, q04, q05, q14, q24]
    device = _create_device(qubits)
    method = greedy.GreedySequenceSearchStrategy('minimal_connectivity')
    assert method.place_line(device, 4) == (q00, q01, q02, q03)
    assert method.place_line(device, 7) == (q00, q01, q02, q03, q04, q14, q24)
    with pytest.raises(NotFoundError):
        _ = method.place_line(device, 8)