from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_cast_none():
    arr = pa.array([1, 2, 3])
    with pytest.raises(TypeError):
        arr.cast(None)