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
def test_dictionary_type():
    ty0 = pa.dictionary(pa.int32(), pa.string())
    assert ty0.index_type == pa.int32()
    assert ty0.value_type == pa.string()
    assert ty0.ordered is False
    ty1 = pa.dictionary(pa.int8(), pa.float64(), ordered=True)
    assert ty1.index_type == pa.int8()
    assert ty1.value_type == pa.float64()
    assert ty1.ordered is True
    ty2 = pa.dictionary('int8', 'string')
    assert ty2.index_type == pa.int8()
    assert ty2.value_type == pa.string()
    assert ty2.ordered is False
    ty3 = pa.dictionary(pa.uint32(), pa.string())
    assert ty3.index_type == pa.uint32()
    assert ty3.value_type == pa.string()
    assert ty3.ordered is False
    with pytest.raises(TypeError):
        pa.dictionary(pa.string(), pa.int64())