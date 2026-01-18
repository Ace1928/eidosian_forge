import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_on_each():
    q0, q1, q2 = cirq.LineQubit.range(3)
    assert cirq.I.on_each(q0, q1, q2) == [cirq.I(q0), cirq.I(q1), cirq.I(q2)]
    assert cirq.I.on_each([q0, [q1], q2]) == [cirq.I(q0), cirq.I(q1), cirq.I(q2)]
    assert cirq.I.on_each(iter([q0, [q1], q2])) == [cirq.I(q0), cirq.I(q1), cirq.I(q2)]
    with pytest.raises(ValueError, match='str'):
        cirq.I.on_each('abc')