import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_non_unitary_eig():
    with pytest.raises(Exception):
        unitary_eig(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]))