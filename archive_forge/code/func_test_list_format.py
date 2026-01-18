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
def test_list_format():
    arr = pa.array([[1], None, [2, 3, None]])
    result = arr.to_string()
    expected = '[\n  [\n    1\n  ],\n  null,\n  [\n    2,\n    3,\n    null\n  ]\n]'
    assert result == expected