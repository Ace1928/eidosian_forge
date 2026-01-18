import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_next_moments_operating_on(circuit_cls):
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = cirq.testing.random_circuit(randint(1, 20), n_moments, random())
        circuit_qubits = circuit.all_qubits()
        n_key_qubits = randint(int(bool(circuit_qubits)), len(circuit_qubits))
        key_qubits = sample(sorted(circuit_qubits), n_key_qubits)
        start = randrange(len(circuit))
        next_moments = circuit.next_moments_operating_on(key_qubits, start)
        for q, m in next_moments.items():
            if m == len(circuit):
                p = circuit.prev_moment_operating_on([q])
            else:
                p = circuit.prev_moment_operating_on([q], m - 1)
            assert not p or p < start