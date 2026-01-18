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
def test_make_rng():
    assert check_random_state(None) is np.random.mtrand._rand
    assert check_random_state(np.random) is np.random.mtrand._rand
    rng_42 = np.random.RandomState(42)
    assert check_random_state(42).randint(100) == rng_42.randint(100)
    rng_42 = np.random.RandomState(42)
    assert check_random_state(rng_42) is rng_42
    rng_42 = np.random.RandomState(42)
    assert check_random_state(43).randint(100) != rng_42.randint(100)
    with pytest.raises(ValueError):
        check_random_state('some invalid seed')