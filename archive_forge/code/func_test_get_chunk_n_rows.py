import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize(('row_bytes', 'max_n_rows', 'working_memory', 'expected'), [(1024, None, 1, 1024), (1024, None, 0.99999999, 1023), (1023, None, 1, 1025), (1025, None, 1, 1023), (1024, None, 2, 2048), (1024, 7, 1, 7), (1024 * 1024, None, 1, 1)])
def test_get_chunk_n_rows(row_bytes, max_n_rows, working_memory, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows, working_memory=working_memory)
    assert actual == expected
    assert type(actual) is type(expected)
    with config_context(working_memory=working_memory):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            actual = get_chunk_n_rows(row_bytes=row_bytes, max_n_rows=max_n_rows)
        assert actual == expected
        assert type(actual) is type(expected)