import pytest
import cirq
def test_default_grouping():
    presorted = (cirq.GridQubit(0, 1), cirq.GridQubit(1, 0), cirq.GridQubit(999, 999), cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(999), cirq.NamedQubit(''), cirq.NamedQubit('0'), cirq.NamedQubit('1'), cirq.NamedQubit('999'), cirq.NamedQubit('a'))
    assert cirq.QubitOrder.DEFAULT.order_for(presorted) == presorted
    assert cirq.QubitOrder.DEFAULT.order_for(reversed(presorted)) == presorted