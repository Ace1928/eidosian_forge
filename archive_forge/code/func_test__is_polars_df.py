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
def test__is_polars_df():
    """Check that _is_polars_df return False for non-dataframe objects."""

    class LooksLikePolars:

        def __init__(self):
            self.columns = ['a', 'b']
            self.schema = ['a', 'b']
    assert not _is_polars_df(LooksLikePolars())