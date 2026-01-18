import pytest
import sympy
import cirq
def test_list_sweep_type_error():
    with pytest.raises(TypeError, match='Not a ParamResolver'):
        _ = cirq.ListSweep([cirq.ParamResolver(), 'bad'])