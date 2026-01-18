from typing import Iterable
import cirq
from cirq_google.line.placement.chip import chip_as_adjacency_list, above, below, right_of, left_of
def test_neighbours():
    qubit = cirq.GridQubit(0, 0)
    assert above(qubit) == cirq.GridQubit(0, -1)
    assert below(qubit) == cirq.GridQubit(0, 1)
    assert right_of(qubit) == cirq.GridQubit(1, 0)
    assert left_of(qubit) == cirq.GridQubit(-1, 0)