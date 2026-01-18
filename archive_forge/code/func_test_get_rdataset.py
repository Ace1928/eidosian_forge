import os
from ssl import SSLError
from socket import timeout
from urllib.error import HTTPError, URLError
import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest
from statsmodels.datasets import get_rdataset, webuse, check_internet, utils
@pytest.mark.smoke
def test_get_rdataset():
    test_url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv'
    internet_available = check_internet(test_url)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        duncan = get_rdataset('Duncan', 'carData', cache=cur_dir)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    assert_(isinstance(duncan, utils.Dataset))
    duncan = get_rdataset('Duncan', 'carData', cache=cur_dir)
    assert_(duncan.from_cache)