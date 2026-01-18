from typing import List
import numpy as np
import pytest
import cirq
def test_merge_k_qubit_unitaries_raises():
    with pytest.raises(ValueError, match='k should be greater than or equal to 1'):
        _ = cirq.merge_k_qubit_unitaries(cirq.Circuit())