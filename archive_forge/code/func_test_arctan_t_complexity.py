import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.algos.mean_estimation.arctan import ArcTan
from cirq_ft.infra.bit_tools import iter_bits_fixed_point
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_arctan_t_complexity():
    gate = ArcTan(4, 5)
    assert cirq_ft.t_complexity(gate) == cirq_ft.TComplexity(t=5)