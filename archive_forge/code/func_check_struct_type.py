import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def check_struct_type(ty, expected):
    """
    Check a struct type is as expected, but not taking order into account.
    """
    assert pa.types.is_struct(ty)
    assert set(ty) == set(expected)