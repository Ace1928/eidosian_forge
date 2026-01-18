from functools import partial
import pytest
from sklearn.datasets.tests.test_common import (
def test_fetch_kddcup99_as_frame(fetch_kddcup99_fxt):
    bunch = fetch_kddcup99_fxt()
    check_as_frame(bunch, fetch_kddcup99_fxt)