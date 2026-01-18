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
def test_dateutil_tzinfo_to_string():
    pytest.importorskip('dateutil')
    import dateutil.tz
    tz = dateutil.tz.UTC
    assert pa.lib.tzinfo_to_string(tz) == 'UTC'
    tz = dateutil.tz.gettz('Europe/Paris')
    assert pa.lib.tzinfo_to_string(tz) == 'Europe/Paris'