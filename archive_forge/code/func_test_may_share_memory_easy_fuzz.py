import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
@pytest.mark.slow
def test_may_share_memory_easy_fuzz():
    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: 1, same_steps=True, min_count=2000)