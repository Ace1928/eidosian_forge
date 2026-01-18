from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_kraus_returns_not_implemented():

    class ReturnsNotImplemented:

        def _kraus_(self):
            return NotImplemented
    assert_not_implemented(ReturnsNotImplemented())