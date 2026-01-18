import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_dict_pauli_like_not_pauli_like():
    p = cirq.MutablePauliString()
    with pytest.raises(TypeError, match='PAULI_GATE_LIKE.*X'):
        p[0] = 1.2