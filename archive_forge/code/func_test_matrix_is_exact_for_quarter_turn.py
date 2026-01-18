from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def test_matrix_is_exact_for_quarter_turn():
    np.testing.assert_equal(cirq.unitary(CExpZinGate(1)), np.diag([1, 1, 1j, -1j]))