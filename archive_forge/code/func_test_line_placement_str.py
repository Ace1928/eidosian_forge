import pytest
import cirq
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_line_placement_str():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    placement = GridQubitLineTuple([q00, q01, q02])
    assert str(placement).strip() == '\nq(0, 0)━━q(0, 1)━━q(0, 2)\n    '.strip()