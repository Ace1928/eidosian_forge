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
def test_concat_array_different_types():
    with pytest.raises(pa.ArrowInvalid):
        pa.concat_arrays([pa.array([1]), pa.array([2.0])])