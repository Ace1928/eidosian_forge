from sympy.ntheory.multinomial import (binomial_coefficients, binomial_coefficients_list, multinomial_coefficients)
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
def test_binomial_coefficients_list():
    assert binomial_coefficients_list(0) == [1]
    assert binomial_coefficients_list(1) == [1, 1]
    assert binomial_coefficients_list(2) == [1, 2, 1]
    assert binomial_coefficients_list(3) == [1, 3, 3, 1]
    assert binomial_coefficients_list(4) == [1, 4, 6, 4, 1]
    assert binomial_coefficients_list(5) == [1, 5, 10, 10, 5, 1]
    assert binomial_coefficients_list(6) == [1, 6, 15, 20, 15, 6, 1]