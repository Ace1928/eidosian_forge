import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.fixture(scope='function', autouse=True)
def no_retry_deprecations():
    with warnings.catch_warnings(record=True) as w:
        yield
    assert len([str(x.message) for x in w if 'Retry' in str(x.message)]) == 0