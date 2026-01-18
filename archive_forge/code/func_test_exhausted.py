import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_exhausted(self):
    assert not Retry(0).is_exhausted()
    assert Retry(-1).is_exhausted()
    assert Retry(1).increment(method='GET').total == 0