import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_block_dispatcher():

    class ArrayLike:
        pass
    a = ArrayLike()
    b = ArrayLike()
    c = ArrayLike()
    assert_equal(list(_block_dispatcher(a)), [a])
    assert_equal(list(_block_dispatcher([a])), [a])
    assert_equal(list(_block_dispatcher([a, b])), [a, b])
    assert_equal(list(_block_dispatcher([[a], [b, [c]]])), [a, b, c])
    assert_equal(list(_block_dispatcher((a, b))), [(a, b)])