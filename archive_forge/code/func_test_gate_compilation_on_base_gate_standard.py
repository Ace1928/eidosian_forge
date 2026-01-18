import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
@pytest.mark.parametrize('tabulation', [sycamore_tabulation, sqrt_iswap_tabulation])
def test_gate_compilation_on_base_gate_standard(tabulation):
    base_gate = tabulation.base_gate
    result = tabulation.compile_two_qubit_gate(base_gate)
    assert len(result.local_unitaries) == 2
    assert result.success
    fidelity = unitary_entanglement_fidelity(result.actual_gate, base_gate)
    assert fidelity > 0.99999