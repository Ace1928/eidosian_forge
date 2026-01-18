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
@classmethod
def bounds_check(cls, el1, el2, msg=None):
    lbrt1 = el1.bounds.lbrt()
    lbrt2 = el2.bounds.lbrt()
    try:
        for v1, v2 in zip(lbrt1, lbrt2):
            if isinstance(v1, datetime_types):
                v1 = dt_to_int(v1)
            if isinstance(v2, datetime_types):
                v2 = dt_to_int(v2)
            cls.assert_array_almost_equal_fn(v1, v2)
    except AssertionError as e:
        raise cls.failureException(f'BoundingBoxes are mismatched: {el1.bounds.lbrt()} != {el2.bounds.lbrt()}.') from e