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
@pickle_test_parametrize
def test_array_pickle_protocol5(data, typ, pickle_module):
    array = pa.array(data, type=typ)
    addresses = [buf.address if buf is not None else 0 for buf in array.buffers()]
    for proto in range(5, pickle_module.HIGHEST_PROTOCOL + 1):
        buffers = []
        pickled = pickle_module.dumps(array, proto, buffer_callback=buffers.append)
        result = pickle_module.loads(pickled, buffers=buffers)
        assert array.equals(result)
        result_addresses = [buf.address if buf is not None else 0 for buf in result.buffers()]
        assert result_addresses == addresses