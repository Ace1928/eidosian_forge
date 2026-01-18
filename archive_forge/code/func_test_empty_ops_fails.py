import cirq
import numpy as np
import pytest
def test_empty_ops_fails():
    ops = []
    with pytest.raises(ValueError, match='must have at least one operation'):
        _ = cirq.KrausChannel(kraus_ops=ops, key='m')