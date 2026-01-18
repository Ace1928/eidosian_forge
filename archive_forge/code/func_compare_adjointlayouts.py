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
def compare_adjointlayouts(cls, el1, el2, msg=None):
    cls.compare_dimensioned(el1, el2)
    for element1, element2 in zip(el1, el1):
        cls.assertEqual(element1, element2)