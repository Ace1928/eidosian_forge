import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
@pytest.mark.parametrize('arg_len', range(4))
def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
    arg_tup = (Any,) * arg_len
    if arg_len in (1, 2):
        assert np.complexfloating[arg_tup]
    else:
        match = f'Too {('few' if arg_len == 0 else 'many')} arguments'
        with pytest.raises(TypeError, match=match):
            np.complexfloating[arg_tup]