import cirq
import numpy as np
import pytest
def test_mix_no_unitaries_fails():
    with pytest.raises(ValueError, match='must have at least one unitary'):
        _ = cirq.MixedUnitaryChannel(mixture=[], key='m')