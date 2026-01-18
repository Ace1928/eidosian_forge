import pytest
import sympy
import cirq
def test_access_sweep():
    sweep = cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6, 7])
    first_elem = sweep[-12]
    assert first_elem == cirq.ParamResolver({'a': 1, 'b': 4})
    sixth_elem = sweep[5]
    assert sixth_elem == cirq.ParamResolver({'a': 2, 'b': 5})