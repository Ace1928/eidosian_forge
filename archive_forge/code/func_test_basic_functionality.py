import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_basic_functionality(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map.items()) == len(pauli_string.items())
    assert set(qubit_pauli_map.items()) == set(pauli_string.items())
    assert len(qubit_pauli_map.values()) == len(pauli_string.values())
    assert set(qubit_pauli_map.values()) == set(pauli_string.values())
    assert len(qubit_pauli_map) == len(pauli_string)
    assert len(qubit_pauli_map.keys()) == len(pauli_string.keys()) == len(pauli_string.qubits)
    assert set(qubit_pauli_map.keys()) == set(pauli_string.keys()) == set(pauli_string.qubits)
    assert len(tuple(qubit_pauli_map)) == len(tuple(pauli_string))
    assert set(tuple(qubit_pauli_map)) == set(tuple(pauli_string))