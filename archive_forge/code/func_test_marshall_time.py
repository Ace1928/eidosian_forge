import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_marshall_time(self):
    now = timeutils.utcnow()
    binary = timeutils.marshall_now(now)
    backagain = timeutils.unmarshall_time(binary)
    self.assertEqual(now, backagain)