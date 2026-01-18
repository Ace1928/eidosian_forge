import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_delta_seconds(self):
    before = timeutils.utcnow()
    after = before + datetime.timedelta(days=7, seconds=59, microseconds=123456)
    self.assertAlmostEqual(604859.123456, timeutils.delta_seconds(before, after))