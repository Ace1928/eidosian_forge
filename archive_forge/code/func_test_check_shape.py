from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
@pytest.mark.parametrize('target,shape_repr,test_shape', [((None,), '(N,)', (1, 3)), ((None, 3), '(N, 3)', (1,)), ((None, 3), '(N, 3)', (1, 2)), ((1, 5), '(1, 5)', (1, 9)), ((None, 2, None), '(M, 2, N)', (1, 3, 1))])
def test_check_shape(target: tuple[int | None, ...], shape_repr: str, test_shape: tuple[int, ...]) -> None:
    error_pattern = '^' + re.escape(f"'aardvark' must be {len(target)}D with shape {shape_repr}, but your input has shape {test_shape}")
    data = np.zeros(test_shape)
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)