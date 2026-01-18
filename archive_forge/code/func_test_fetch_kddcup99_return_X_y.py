from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
def test_fetch_kddcup99_return_X_y(fetch_kddcup99_fxt):
    fetch_func = partial(fetch_kddcup99_fxt, subset='smtp')
    data = fetch_func()
    check_return_X_y(data, fetch_func)