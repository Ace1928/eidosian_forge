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
def test_dictionary_from_numpy():
    indices = np.repeat([0, 1, 2], 2)
    dictionary = np.array(['foo', 'bar', 'baz'], dtype=object)
    mask = np.array([False, False, True, False, False, False])
    d1 = pa.DictionaryArray.from_arrays(indices, dictionary)
    d2 = pa.DictionaryArray.from_arrays(indices, dictionary, mask=mask)
    assert d1.indices.to_pylist() == indices.tolist()
    assert d1.indices.to_pylist() == indices.tolist()
    assert d1.dictionary.to_pylist() == dictionary.tolist()
    assert d2.dictionary.to_pylist() == dictionary.tolist()
    for i in range(len(indices)):
        assert d1[i].as_py() == dictionary[indices[i]]
        if mask[i]:
            assert d2[i].as_py() is None
        else:
            assert d2[i].as_py() == dictionary[indices[i]]