import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_sums_uint8(self):
    if sys.platform == 'darwin' and platform.machine() == 'x86_64' or USING_CLANG_CL:
        pytest.xfail('Fails on macOS x86-64 and when using clang-cl with Meson, see gh-23838')
    self.check_einsum_sums('u1')