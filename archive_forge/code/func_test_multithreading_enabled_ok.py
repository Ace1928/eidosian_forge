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
@pytest.mark.parametrize('args,kwargs', [((np.empty((1,), dtype=float),), {}), ((), {'a': np.empty((1,), dtype=float)}), (([1],), {}), ((), {'a': [1]}), ((), {'out': np.empty((1,), dtype=object)}), ((), {'where': np.empty((1,), dtype=object)})])
def test_multithreading_enabled_ok(args, kwargs):
    result = set_first_element(42, *args, **kwargs)
    assert result[0] == 42