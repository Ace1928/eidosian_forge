import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_floor_exact_16():
    assert floor_exact(2 ** 31, np.float16) == np.inf
    assert floor_exact(-2 ** 31, np.float16) == -np.inf