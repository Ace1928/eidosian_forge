from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_clears_paired_cnot():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(before=cirq.Circuit(cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.CNOT(a, b))), expected=cirq.Circuit())