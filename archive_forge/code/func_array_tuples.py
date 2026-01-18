from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@pytest.fixture(scope='module')
def array_tuples(arrays):
    return [(arrays[0], arrays[0]), (arrays[0], arrays[1]), (arrays[1], arrays[1]), (arrays[2], arrays[2]), (arrays[2], arrays[3]), (arrays[2], arrays[4]), (arrays[4], arrays[2]), (arrays[3], arrays[3]), (arrays[4], arrays[4])]