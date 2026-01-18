import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n', [*range(3, 10)])
@allow_deprecated_cirq_ft_use_in_tests
def test_addition_gate_t_complexity(n: int):
    g = cirq_ft.AdditionGate(n)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)
    cirq.testing.assert_equivalent_repr(g, setup_code='import cirq_ft')