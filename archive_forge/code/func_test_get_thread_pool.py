from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
@mock.patch('glance.async_.get_threadpool_model')
def test_get_thread_pool(self, mock_gtm):
    get_thread_pool = glance.api.common.get_thread_pool
    pool1 = get_thread_pool('pool1', size=123)
    get_thread_pool('pool2', size=456)
    pool1a = get_thread_pool('pool1')
    self.assertEqual(pool1, pool1a)
    mock_gtm.return_value.assert_has_calls([mock.call(123), mock.call(456)])