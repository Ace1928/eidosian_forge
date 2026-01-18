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
@pytest.mark.parametrize('key', [['col1'], ['col2'], ['col1', 'col2'], ['col1', 'col3'], ['col2', 'col3']])
def test_get_column_indices_pandas_nonunique_columns_error(key):
    pd = pytest.importorskip('pandas')
    toy = np.zeros((1, 5), dtype=int)
    columns = ['col1', 'col1', 'col2', 'col3', 'col2']
    X = pd.DataFrame(toy, columns=columns)
    err_msg = 'Selected columns, {}, are not unique in dataframe'.format(key)
    with pytest.raises(ValueError) as exc_info:
        _get_column_indices(X, key)
    assert str(exc_info.value) == err_msg