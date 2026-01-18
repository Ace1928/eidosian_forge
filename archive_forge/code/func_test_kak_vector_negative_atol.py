import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_vector_negative_atol():
    with pytest.raises(ValueError, match='must be positive'):
        cirq.kak_vector(np.eye(4), atol=-1.0)