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
def test_circuit_length_inference():
    circuit = cirq.Circuit(cirq.X(cirq.q(0)))
    qubit_indices = {cirq.q(0): 0}
    mkey_indices = {}
    ckey_indices = {}
    assert circuits.circuit.get_earliest_accommodating_moment_index(cirq.Moment(), qubit_indices, mkey_indices, ckey_indices) == len(circuit)