import pytest
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
import hypothesis.extra.numpy as npst
from scipy.special._support_alternative_backends import (get_array_special_func,
from scipy.conftest import array_api_compatible
from scipy import special
from scipy._lib._array_api import xp_assert_close
from scipy._lib.array_api_compat.array_api_compat import numpy as np
import numpy.array_api as np_array_api
def test_dispatch_to_unrecognize_library():
    xp = np_array_api
    f = get_array_special_func('ndtr', xp=xp, n_array_args=1)
    x = [1, 2, 3]
    res = f(xp.asarray(x))
    ref = xp.asarray(special.ndtr(np.asarray(x)))
    xp_assert_close(res, ref)