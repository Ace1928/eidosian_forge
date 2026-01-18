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
def check_run_end_encoded_from_arrays_with_type(ree_type=None):
    run_ends = [3, 5, 10, 19]
    values = [1, 2, 1, 3]
    ree_array = pa.RunEndEncodedArray.from_arrays(run_ends, values, ree_type)
    check_run_end_encoded(ree_array, run_ends, values, 19, 4, 0)