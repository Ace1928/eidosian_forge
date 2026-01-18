import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test__create(self):
    manager = base.Manager()
    manager.api = mock.Mock()
    manager.api.client = mock.Mock()
    response_key = 'response_key'
    data_ = 'test-data'
    body_ = {response_key: data_}
    url_ = 'test_url_post'
    manager.api.client.post = mock.Mock(return_value=(url_, body_))
    return_raw = True
    r = manager._create(url_, body_, response_key, return_raw)
    self.assertEqual(data_, r)
    return_raw = False

    @contextlib.contextmanager
    def completion_cache_mock(*arg, **kwargs):
        yield
    mockl = mock.Mock()
    mockl.side_effect = completion_cache_mock
    manager.completion_cache = mockl
    manager.resource_class = mock.Mock(return_value='test-class')
    r = manager._create(url_, body_, response_key, return_raw)
    self.assertEqual('test-class', r)