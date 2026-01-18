import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_assert_decompose_is_consistent_with_t_complexity():
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(cirq.T)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(DoesNotDecompose())
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(cirq_ft.testing.GateHelper(cirq_ft.And()).operation)