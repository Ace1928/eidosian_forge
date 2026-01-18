import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_flop_cost():
    size_dict = {v: 10 for v in 'abcdef'}
    assert 10 == oe.helpers.flop_count('a', False, 1, size_dict)
    assert 10 == oe.helpers.flop_count('a', False, 2, size_dict)
    assert 100 == oe.helpers.flop_count('ab', False, 2, size_dict)
    assert 20 == oe.helpers.flop_count('a', True, 2, size_dict)
    assert 200 == oe.helpers.flop_count('ab', True, 2, size_dict)
    assert 30 == oe.helpers.flop_count('a', True, 3, size_dict)
    assert 2000 == oe.helpers.flop_count('abc', True, 2, size_dict)