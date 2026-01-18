import os
import sys
from inspect import cleandoc
from itertools import chain
from string import ascii_letters, digits
from unittest import mock
import numpy as np
import pytest
import shapely
from shapely.decorators import multithreading_enabled, requires_geos
def test_multithreading_enabled_preserves_flag():
    arr = np.empty((1,), dtype=object)
    arr.flags.writeable = False
    with pytest.raises(ValueError):
        set_first_element(42, arr)
    assert not arr.flags.writeable