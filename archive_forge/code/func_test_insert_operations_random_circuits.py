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
@pytest.mark.parametrize('circuit', [cirq.testing.random_circuit(cirq.LineQubit.range(10), 10, 0.5) for _ in range(20)])
def test_insert_operations_random_circuits(circuit):
    n_moments = len(circuit)
    operations, insert_indices = ([], [])
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            operations.append(op)
            insert_indices.append(moment_index)
    other_circuit = cirq.Circuit([cirq.Moment() for _ in range(n_moments)])
    other_circuit._insert_operations(operations, insert_indices)
    assert circuit == other_circuit