import pytest
import sympy
import cirq
def test_dict_to_product_sweep():
    assert cirq.dict_to_product_sweep({'t': [0, 2, 3]}) == cirq.Product(cirq.Points('t', [0, 2, 3]))
    assert cirq.dict_to_product_sweep({'t': [0, 1], 's': [2, 3], 'r': 4}) == cirq.Product(cirq.Points('t', [0, 1]), cirq.Points('s', [2, 3]), cirq.Points('r', [4]))