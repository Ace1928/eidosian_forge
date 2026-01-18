from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_generic_select_raises():
    with pytest.raises(ValueError, match='should contain 3'):
        _ = cirq_ft.GenericSelect(2, 3, [cirq.DensePauliString('Y')])
    with pytest.raises(ValueError, match='should be at-least 3'):
        _ = cirq_ft.GenericSelect(1, 2, [cirq.DensePauliString('XX')] * 5)