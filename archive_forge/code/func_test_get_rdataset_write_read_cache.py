import os
from ssl import SSLError
from socket import timeout
from urllib.error import HTTPError, URLError
import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest
from statsmodels.datasets import get_rdataset, webuse, check_internet, utils
@pytest.mark.smoke
def test_get_rdataset_write_read_cache():
    try:
        guerry = get_rdataset('Guerry', 'HistData', cache=cur_dir)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    assert_(guerry.from_cache is False)
    guerry2 = get_rdataset('Guerry', 'HistData', cache=cur_dir)
    assert_(guerry2.from_cache is True)
    fn = 'raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,csv,HistData,Guerry-v2.csv.zip'
    os.remove(os.path.join(cur_dir, fn))
    fn = 'raw.githubusercontent.com,vincentarelbundock,Rdatasets,master,doc,HistData,rst,Guerry-v2.rst.zip'
    os.remove(os.path.join(cur_dir, fn))