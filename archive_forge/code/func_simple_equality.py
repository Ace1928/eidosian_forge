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
def simple_equality(cls, first, second, msg=None):
    """
        Classmethod equivalent to unittest.TestCase method (longMessage = False.)
        """
    check = first == second
    if not isinstance(check, bool) and hasattr(check, 'all'):
        check = check.all()
    if not check:
        standardMsg = f'{safe_repr(first)} != {safe_repr(second)}'
        raise cls.failureException(msg or standardMsg)