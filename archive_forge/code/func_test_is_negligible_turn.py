import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_is_negligible_turn():
    assert cirq.is_negligible_turn(0, 1e-05)
    assert cirq.is_negligible_turn(1e-06, 1e-05)
    assert cirq.is_negligible_turn(1, 1e-05)
    assert cirq.is_negligible_turn(1 + 1e-06, 1e-05)
    assert cirq.is_negligible_turn(1 - 1e-06, 1e-05)
    assert cirq.is_negligible_turn(-1, 1e-05)
    assert cirq.is_negligible_turn(-1 + 1e-06, 1e-05)
    assert cirq.is_negligible_turn(-1 - 1e-06, 1e-05)
    assert cirq.is_negligible_turn(3, 1e-05)
    assert cirq.is_negligible_turn(3 + 1e-06, 1e-05)
    assert not cirq.is_negligible_turn(0.0001, 1e-05)
    assert not cirq.is_negligible_turn(-0.0001, 1e-05)
    assert not cirq.is_negligible_turn(0.5, 1e-05)
    assert not cirq.is_negligible_turn(-0.5, 1e-05)
    assert not cirq.is_negligible_turn(0.5, 1e-05)
    assert not cirq.is_negligible_turn(4.5, 1e-05)
    assert not cirq.is_negligible_turn(sympy.Symbol('a'), 1e-05)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') + 1, 1e-05)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') * 1e-10, 1e-05)
    assert cirq.is_negligible_turn(sympy.Symbol('a') * 0 + 3 + 1e-06, 1e-05)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') * 0 + 1.5 - 1e-06, 1e-05)