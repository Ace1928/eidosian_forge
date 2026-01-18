import numpy as np
import pytest
import sympy
import cirq
def test_canonicalize_half_turns():
    assert cirq.canonicalize_half_turns(0) == 0
    assert cirq.canonicalize_half_turns(1) == +1
    assert cirq.canonicalize_half_turns(-1) == +1
    assert cirq.canonicalize_half_turns(0.5) == 0.5
    assert cirq.canonicalize_half_turns(1.5) == -0.5
    assert cirq.canonicalize_half_turns(-0.5) == -0.5
    assert cirq.canonicalize_half_turns(101.5) == -0.5
    assert cirq.canonicalize_half_turns(sympy.Symbol('a')) == sympy.Symbol('a')
    assert cirq.canonicalize_half_turns(sympy.Symbol('a') + 1) == sympy.Symbol('a') + 1
    assert cirq.canonicalize_half_turns(sympy.Symbol('a') * 0 + 3) == 1