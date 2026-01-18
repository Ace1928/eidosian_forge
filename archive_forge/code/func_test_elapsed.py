import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_elapsed(self, mock_now):
    mock_now.side_effect = monotonic_iter(incr=0.2)
    watch = timeutils.StopWatch()
    watch.start()
    matcher = matchers.GreaterThan(0.19)
    self.assertThat(watch.elapsed(), matcher)