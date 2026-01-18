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
def test_multithreaded_sharing():
    from multiprocessing.pool import ThreadPool

    def fn():
        X, Y, Z = helpers.build_views('ab,bc,cd')
        with shared_intermediates():
            contract('ab,bc,cd->a', X, Y, Z)
            contract('ab,bc,cd->b', X, Y, Z)
            return len(get_sharing_cache())
    expected = fn()
    pool = ThreadPool(8)
    fs = [pool.apply_async(fn) for _ in range(16)]
    assert not currently_sharing()
    assert [f.get() for f in fs] == [expected] * 16
    pool.close()