import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_equal_up_to_global_phase_on_diff_types():
    op = cirq.X(cirq.LineQubit(0))
    assert not cirq.equal_up_to_global_phase(op, 3)