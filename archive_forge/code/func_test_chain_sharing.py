import itertools
import weakref
from collections import Counter
import numpy as np
import pytest
from opt_einsum import (contract, contract_expression, contract_path, get_symbol, helpers, shared_intermediates)
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import (count_cached_ops, currently_sharing, get_sharing_cache)
@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('backend', backends)
def test_chain_sharing(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    alphabet = ''.join((get_symbol(i) for i in range(size + 1)))
    names = [alphabet[i:i + 2] for i in range(size)]
    inputs = ','.join(names)
    num_exprs_nosharing = 0
    for i in range(size + 1):
        with shared_intermediates() as cache:
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
            num_exprs_nosharing += _compute_cost(cache)
    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
        num_exprs_sharing = _compute_cost(cache)
    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing