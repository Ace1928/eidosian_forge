from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('cls', [HasKraus, HasMixture, HasUnitary])
def test_has_kraus(cls):
    assert cirq.has_kraus(cls())