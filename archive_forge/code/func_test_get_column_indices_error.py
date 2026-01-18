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
@pytest.mark.parametrize('key, err_msg', [(10, 'all features must be in \\[0, 2\\]'), ('whatever', 'A given column is not a column of the dataframe'), (object(), 'No valid specification of the columns')])
def test_get_column_indices_error(key, err_msg):
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame(X_toy, columns=['col_0', 'col_1', 'col_2'])
    with pytest.raises(ValueError, match=err_msg):
        _get_column_indices(X_df, key)