import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
@pytest.mark.parametrize('optimize', ['greedy', 'branch-2', 'branch-all', 'optimal', 'dp'])
def test_can_optimize_outer_products(optimize):
    a, b, c = [np.random.randn(10, 10) for _ in range(3)]
    d = np.random.randn(10, 2)
    assert oe.contract_path('ab,cd,ef,fg', a, b, c, d, optimize=optimize)[0] == [(2, 3), (0, 2), (0, 1)]