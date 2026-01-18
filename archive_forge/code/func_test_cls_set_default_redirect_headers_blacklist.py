import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_cls_set_default_redirect_headers_blacklist(self, expect_retry_deprecation):
    old_setting = Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST
    try:
        Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST = {'test'}
        retry = Retry()
        assert retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT == {'test'}
        assert retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST == {'test'}
        assert retry.remove_headers_on_redirect == {'test'}
        assert retry.remove_headers_on_redirect == {'test'}
        retry = Retry(remove_headers_on_redirect={'test2'})
        assert retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT == {'test'}
        assert retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST == {'test'}
        assert retry.remove_headers_on_redirect == {'test2'}
        assert retry.remove_headers_on_redirect == {'test2'}
    finally:
        Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST = old_setting
        assert Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST == old_setting