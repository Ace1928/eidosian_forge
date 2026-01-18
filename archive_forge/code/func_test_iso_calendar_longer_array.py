from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_iso_calendar_longer_array(unit):
    arr = pa.array([datetime.datetime(2022, 1, 2, 9)] * 50, pa.timestamp(unit))
    result = pc.iso_calendar(arr)
    expected = pa.StructArray.from_arrays([[2021] * 50, [52] * 50, [7] * 50], names=['iso_year', 'iso_week', 'iso_day_of_week'])
    assert result.equals(expected)