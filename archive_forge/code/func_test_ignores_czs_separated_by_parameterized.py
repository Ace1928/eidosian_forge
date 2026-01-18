from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_ignores_czs_separated_by_parameterized():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(before=cirq.Circuit([cirq.Moment(cirq.CZ(a, b)), cirq.Moment(cirq.Z(a) ** sympy.Symbol('boo')), cirq.Moment(cirq.CZ(a, b))]), expected=cirq.Circuit([cirq.Moment(cirq.CZ(a, b)), cirq.Moment(cirq.Z(a) ** sympy.Symbol('boo')), cirq.Moment(cirq.CZ(a, b))]), additional_gates=[cirq.ZPowGate])