import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
def test_gate_compilation_missing_points_raises_error():
    with pytest.raises(ValueError, match='Failed to tabulate a'):
        two_qubit_gate_product_tabulation(np.eye(4), 0.4, allow_missed_points=False, random_state=_rng)