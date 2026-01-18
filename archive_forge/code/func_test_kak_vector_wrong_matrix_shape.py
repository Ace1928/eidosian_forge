import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('bad_input', cases)
def test_kak_vector_wrong_matrix_shape(bad_input):
    with pytest.raises(ValueError, match='to have shape'):
        cirq.kak_vector(bad_input)