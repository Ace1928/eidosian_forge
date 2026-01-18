import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_add_mod_n_protocols():
    with pytest.raises(ValueError, match='must be between'):
        _ = cirq_ft.AddMod(3, 10)
    add_one = cirq_ft.AddMod(3, 5, 1)
    add_two = cirq_ft.AddMod(3, 5, 2, cv=[1, 0])
    assert add_one == cirq_ft.AddMod(3, 5, 1)
    assert add_one != add_two
    assert hash(add_one) != hash(add_two)
    assert add_two.cv == (1, 0)
    assert cirq.circuit_diagram_info(add_two).wire_symbols == ('@', '@(0)') + ('Add_2_Mod_5',) * 3