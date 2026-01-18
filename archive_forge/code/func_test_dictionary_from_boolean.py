import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_dictionary_from_boolean():
    typ = pa.dictionary(pa.int8(), value_type=pa.bool_())
    a = pa.array([False, False, True, False, True], type=typ)
    assert isinstance(a.type, pa.DictionaryType)
    assert a.type.equals(typ)
    expected_indices = pa.array([0, 0, 1, 0, 1], type=pa.int8())
    expected_dictionary = pa.array([False, True], type=pa.bool_())
    assert a.indices.equals(expected_indices)
    assert a.dictionary.equals(expected_dictionary)