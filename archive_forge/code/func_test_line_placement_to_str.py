import pytest
import cirq
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_line_placement_to_str():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    placement = GridQubitLineTuple([q02, q01, q00, q10, q11])
    assert str(placement).strip() == '\nq(0, 0)━━q(0, 1)━━q(0, 2)\n┃\nq(1, 0)━━q(1, 1)\n    '.strip()