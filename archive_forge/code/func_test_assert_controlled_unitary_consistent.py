from typing import Optional, Sequence, Union, Collection, Tuple, List
import pytest
import numpy as np
import cirq
from cirq.ops import control_values as cv
def test_assert_controlled_unitary_consistent():
    cirq.testing.assert_controlled_and_controlled_by_identical(GoodGate(exponent=0.5, global_shift=1 / 3))
    with pytest.raises(AssertionError):
        cirq.testing.assert_controlled_and_controlled_by_identical(BadGate(exponent=0.5, global_shift=1 / 3))