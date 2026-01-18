import itertools
import random
from typing import List, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('cv, adjoint, str_output', [((1, 1, 1), False, 'And'), ((1, 0, 1), False, 'And(1, 0, 1)'), ((1, 1, 1), True, 'And†'), ((1, 0, 1), True, 'And†(1, 0, 1)')])
@allow_deprecated_cirq_ft_use_in_tests
def test_and_gate_str_and_repr(cv, adjoint, str_output):
    gate = cirq_ft.And(cv, adjoint=adjoint)
    assert str(gate) == str_output
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft\n')