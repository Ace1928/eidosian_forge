import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_cls_get_default_method_whitelist(self, expect_retry_deprecation):
    assert Retry.DEFAULT_ALLOWED_METHODS == Retry.DEFAULT_METHOD_WHITELIST