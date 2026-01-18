import math
import random
import pytest
from cirq_ft.infra.bit_tools import (
def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(0, 3, signed=True)) == [0, 0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(1, 2, signed=True)) == [0, 1]
    assert list(iter_bits(-1, 2, signed=True)) == [1, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(2, 3, signed=True)) == [0, 1, 0]
    assert list(iter_bits(-2, 3, signed=True)) == [1, 1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]
    with pytest.raises(ValueError):
        _ = list(iter_bits(-3, 4))