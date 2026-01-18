from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
def test_gae_environ():
    assert not appengine.is_appengine()
    assert not appengine.is_appengine_sandbox()
    assert not appengine.is_local_appengine()
    assert not appengine.is_prod_appengine()
    assert not appengine.is_prod_appengine_mvms()