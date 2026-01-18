from __future__ import annotations
from typing import Callable
import pytest
from itertools import product
from numpy.testing import assert_allclose, suppress_warnings
from scipy import special
from scipy.special import cython_special
def test_cython_api_completeness():
    for name in dir(cython_special):
        func = getattr(cython_special, name)
        if callable(func) and (not name.startswith('_')):
            for _, cyfun, _, _ in PARAMS:
                if cyfun is func:
                    break
            else:
                raise RuntimeError(f'{name} missing from tests!')