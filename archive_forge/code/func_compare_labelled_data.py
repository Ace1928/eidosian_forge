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
def compare_labelled_data(cls, obj1, obj2, msg=None):
    cls.assertEqual(obj1.group, obj2.group, 'Group labels mismatched.')
    cls.assertEqual(obj1.label, obj2.label, 'Labels mismatched.')