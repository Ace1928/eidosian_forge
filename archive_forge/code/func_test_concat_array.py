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
def test_concat_array():
    concatenated = pa.concat_arrays([pa.array([1, 2]), pa.array([3, 4])])
    assert concatenated.equals(pa.array([1, 2, 3, 4]))