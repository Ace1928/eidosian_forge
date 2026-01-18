from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def test_choi_to_kraus_atol():
    """Verifies that insignificant Kraus operators are omitted."""
    choi = cirq.kraus_to_choi(cirq.kraus(cirq.phase_damp(1e-06)))
    assert len(cirq.choi_to_kraus(choi, atol=0.01)) == 1
    assert len(cirq.choi_to_kraus(choi, atol=0.0001)) == 2