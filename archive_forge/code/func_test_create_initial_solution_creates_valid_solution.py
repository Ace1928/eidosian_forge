from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_create_initial_solution_creates_valid_solution():

    def check_chip(qubits: List[cirq.GridQubit]):
        _verify_valid_state(qubits, AnnealSequenceSearch(_create_device(qubits), seed=4027383825)._create_initial_solution())
    q00, q01, q02 = [cirq.GridQubit(0, x) for x in range(3)]
    q10, q11, q12 = [cirq.GridQubit(1, x) for x in range(3)]
    check_chip([q00, q10])
    check_chip([q00, q10, q01])
    check_chip([q00, q10, q01])
    check_chip([q00, q10, q01, q11])
    check_chip([q00, q10, q02, q12])
    check_chip([q00, q10, q11, q02])
    check_chip([q00, q10, q02])
    check_chip([q00, q10, q01, q11, q02, q12])