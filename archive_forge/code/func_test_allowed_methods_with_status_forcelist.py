import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_allowed_methods_with_status_forcelist(self):
    retry = Retry(status_forcelist=[500], allowed_methods=None)
    assert retry.is_retry('GET', status_code=500)
    assert retry.is_retry('POST', status_code=500)
    retry = Retry(status_forcelist=[500], allowed_methods=['POST'])
    assert not retry.is_retry('GET', status_code=500)
    assert retry.is_retry('POST', status_code=500)