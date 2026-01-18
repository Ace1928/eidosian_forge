from scipy.datasets._registry import registry
from scipy.datasets._fetchers import data_fetcher
from scipy.datasets._utils import _clear_cache
from scipy.datasets import ascent, face, electrocardiogram, download_all
from numpy.testing import assert_equal, assert_almost_equal
import os
import pytest
@pytest.fixture(scope='module', autouse=True)
def test_download_all(self):
    download_all()
    yield