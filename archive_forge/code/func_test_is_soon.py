import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_is_soon(self):
    expires = timeutils.utcnow() + datetime.timedelta(minutes=5)
    self.assertFalse(timeutils.is_soon(expires, 120))
    self.assertTrue(timeutils.is_soon(expires, 300))
    self.assertTrue(timeutils.is_soon(expires, 600))
    with mock.patch('datetime.datetime') as datetime_mock:
        datetime_mock.now.return_value = self.skynet_self_aware_time
        expires = timeutils.utcnow()
        self.assertTrue(timeutils.is_soon(expires, 0))