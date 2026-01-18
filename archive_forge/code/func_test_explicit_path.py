import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_explicit_path():
    x = oe.contract('a,b,c', [1], [2], [3], optimize=[(1, 2), (0, 1)])
    assert x.item() == 6