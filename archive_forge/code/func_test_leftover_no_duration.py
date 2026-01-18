import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_leftover_no_duration(self):
    watch = timeutils.StopWatch()
    watch.start()
    self.assertRaises(RuntimeError, watch.leftover)
    self.assertRaises(RuntimeError, watch.leftover, return_none=False)
    self.assertIsNone(watch.leftover(return_none=True))