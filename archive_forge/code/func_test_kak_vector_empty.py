import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_vector_empty():
    assert len(cirq.kak_vector([])) == 0