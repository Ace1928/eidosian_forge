from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_duration():
    for unit in ('s', 'ms', 'us', 'ns'):
        ty = pa.duration(unit)
        assert ty.unit == unit
    for invalid_unit in ('m', 'arbit', 'rary'):
        with pytest.raises(ValueError, match='Invalid time unit'):
            pa.duration(invalid_unit)