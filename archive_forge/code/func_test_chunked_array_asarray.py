from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
@pytest.mark.nopandas
def test_chunked_array_asarray():
    data = [pa.array([0]), pa.array([1, 2, 3])]
    chunked_arr = pa.chunked_array(data)
    np_arr = np.asarray(chunked_arr)
    assert np_arr.tolist() == [0, 1, 2, 3]
    assert np_arr.dtype == np.dtype('int64')
    np_arr = np.asarray(chunked_arr, dtype='str')
    assert np_arr.tolist() == ['0', '1', '2', '3']
    data = [pa.array([1, None]), pa.array([1, 2, 3])]
    chunked_arr = pa.chunked_array(data)
    np_arr = np.asarray(chunked_arr)
    elements = np_arr.tolist()
    assert elements[0] == 1.0
    assert np.isnan(elements[1])
    assert elements[2:] == [1.0, 2.0, 3.0]
    assert np_arr.dtype == np.dtype('float64')
    arr = pa.DictionaryArray.from_arrays(pa.array([0, 1, 2, 0, 1]), pa.array(['a', 'b', 'c']))
    chunked_arr = pa.chunked_array([arr, arr])
    np_arr = np.asarray(chunked_arr)
    assert np_arr.dtype == np.dtype('object')
    assert np_arr.tolist() == ['a', 'b', 'c', 'a', 'b'] * 2