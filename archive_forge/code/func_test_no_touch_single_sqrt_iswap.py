from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_no_touch_single_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.Moment([cirq.ISwapPowGate(exponent=0.5, global_shift=-0.5).on(a, b).with_tags('mytag')])])
    assert_optimizes(before=circuit, expected=circuit)