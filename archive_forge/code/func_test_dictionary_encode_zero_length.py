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
def test_dictionary_encode_zero_length():
    arr = pa.array([], type=pa.string())
    encoded = arr.dictionary_encode()
    assert len(encoded.dictionary) == 0
    encoded.validate(full=True)