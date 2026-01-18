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
@pytest.mark.parametrize('key, dtype', [(0, 'int'), ('0', 'str'), (True, 'bool'), (np.bool_(True), 'bool'), ([0, 1, 2], 'int'), (['0', '1', '2'], 'str'), ((0, 1, 2), 'int'), (('0', '1', '2'), 'str'), (slice(None, None), None), (slice(0, 2), 'int'), (np.array([0, 1, 2], dtype=np.int32), 'int'), (np.array([0, 1, 2], dtype=np.int64), 'int'), (np.array([0, 1, 2], dtype=np.uint8), 'int'), ([True, False], 'bool'), ((True, False), 'bool'), (np.array([True, False]), 'bool'), ('col_0', 'str'), (['col_0', 'col_1', 'col_2'], 'str'), (('col_0', 'col_1', 'col_2'), 'str'), (slice('begin', 'end'), 'str'), (np.array(['col_0', 'col_1', 'col_2']), 'str'), (np.array(['col_0', 'col_1', 'col_2'], dtype=object), 'str')])
def test_determine_key_type(key, dtype):
    assert _determine_key_type(key) == dtype