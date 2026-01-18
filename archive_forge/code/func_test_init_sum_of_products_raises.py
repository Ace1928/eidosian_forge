import cirq
import pytest
def test_init_sum_of_products_raises():
    with pytest.raises(ValueError):
        _ = cirq.SumOfProducts([])
    with pytest.raises(ValueError):
        _ = cirq.SumOfProducts([[1], [1, 0]])