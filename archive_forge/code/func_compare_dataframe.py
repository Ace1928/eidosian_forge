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
def compare_dataframe(cls, df1, df2, msg='DFrame'):
    from pandas.testing import assert_frame_equal
    try:
        assert_frame_equal(df1, df2)
    except AssertionError as e:
        raise cls.failureException(f'{msg}: {e}') from e