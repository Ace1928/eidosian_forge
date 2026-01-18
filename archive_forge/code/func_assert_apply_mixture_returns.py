from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def assert_apply_mixture_returns(val: Any, rho: np.ndarray, left_axes: Iterable[int], right_axes: Optional[Iterable[int]], assert_result_is_out_buf: bool=False, expected_result: Optional[np.ndarray]=None):
    out_buf, buf0, buf1 = make_buffers(rho.shape, rho.dtype)
    result = cirq.apply_mixture(val, args=cirq.ApplyMixtureArgs(target_tensor=rho, left_axes=left_axes, right_axes=right_axes, out_buffer=out_buf, auxiliary_buffer0=buf0, auxiliary_buffer1=buf1))
    if assert_result_is_out_buf:
        assert result is out_buf
    else:
        assert result is not out_buf
    assert expected_result is not None
    np.testing.assert_array_almost_equal(result, expected_result)