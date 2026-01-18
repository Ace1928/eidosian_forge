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
def check_run_end_encoded(ree_array, run_ends, values, logical_length, physical_length, physical_offset):
    assert ree_array.run_ends.to_pylist() == run_ends
    assert ree_array.values.to_pylist() == values
    assert len(ree_array) == logical_length
    assert ree_array.find_physical_length() == physical_length
    assert ree_array.find_physical_offset() == physical_offset