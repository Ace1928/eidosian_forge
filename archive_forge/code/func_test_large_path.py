import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
@pytest.mark.parametrize('num_symbols', [2, 3, 26, 26 + 26, 256 - 140, 300])
def test_large_path(num_symbols):
    symbols = ''.join((oe.get_symbol(i) for i in range(num_symbols)))
    dimension_dict = dict(zip(symbols, itertools.cycle([2, 3, 4])))
    expression = ','.join((symbols[t:t + 2] for t in range(num_symbols - 1)))
    tensors = oe.helpers.build_views(expression, dimension_dict=dimension_dict)
    oe.contract_path(expression, *tensors, optimize='greedy')