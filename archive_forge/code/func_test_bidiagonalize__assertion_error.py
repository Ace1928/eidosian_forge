import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def test_bidiagonalize__assertion_error():
    with pytest.raises(AssertionError):
        a = np.diag([0, 1])
        assert_bidiagonalized_by(a, a, a)