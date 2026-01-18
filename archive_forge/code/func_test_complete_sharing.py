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
@pytest.mark.parametrize('backend', backends)
def test_complete_sharing(backend):
    eq = 'ab,bc,cd->'
    views = helpers.build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))
    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)
    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)
    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected