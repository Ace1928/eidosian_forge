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
def compare_itemtables(cls, el1, el2, msg=None):
    cls.compare_dimensioned(el1, el2)
    if el1.rows != el2.rows:
        raise cls.failureException('ItemTables have different numbers of rows.')
    if el1.cols != el2.cols:
        raise cls.failureException('ItemTables have different numbers of columns.')
    if [d.name for d in el1.vdims] != [d.name for d in el2.vdims]:
        raise cls.failureException('ItemTables have different Dimensions.')