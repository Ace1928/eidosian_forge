from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_acquaintance_gate_unknown_qubit_count():
    assert cirq.circuit_diagram_info(cca.acquaint, default=None) is None