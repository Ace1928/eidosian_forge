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
def test_key_value_metadata_duplicates():
    meta = pa.KeyValueMetadata({'a': '1', 'b': '2'})
    with pytest.raises(KeyError):
        pa.KeyValueMetadata(meta, a='3')