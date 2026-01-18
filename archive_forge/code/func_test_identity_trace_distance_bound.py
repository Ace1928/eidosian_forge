import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_trace_distance_bound():
    assert cirq.I._trace_distance_bound_() == 0
    assert cirq.IdentityGate(num_qubits=2)._trace_distance_bound_() == 0