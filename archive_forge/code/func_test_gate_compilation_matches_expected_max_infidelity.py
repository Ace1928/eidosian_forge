import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
@pytest.mark.parametrize('tabulation', [sycamore_tabulation, sqrt_iswap_tabulation])
@pytest.mark.parametrize('target', _random_2Q_unitaries)
def test_gate_compilation_matches_expected_max_infidelity(tabulation, target):
    result = tabulation.compile_two_qubit_gate(target)
    assert result.success
    max_error = tabulation.max_expected_infidelity
    assert 1 - unitary_entanglement_fidelity(target, result.actual_gate) < max_error