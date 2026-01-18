import contextlib
from functools import partial
from unittest import TestCase
from unittest.util import safe_repr
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..core import (
from ..core.options import Cycle, Options
from ..core.util import cast_array_to_int64, datetime_types, dt_to_int, is_float
from . import *  # noqa (All Elements need to support comparison)
def assert_element_equal(element1, element2):
    hv_types = (Element, Layout)
    if not isinstance(element1, hv_types):
        raise TypeError(f'First argument is not an allowed type but a {type(element1).__name__!r}.')
    if not isinstance(element2, hv_types):
        raise TypeError(f'Second argument is not an allowed type but a {type(element2).__name__!r}.')
    _assert_element_equal(element1, element2)