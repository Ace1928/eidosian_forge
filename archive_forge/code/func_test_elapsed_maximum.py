import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_elapsed_maximum(self, mock_now):
    mock_now.side_effect = [0, 1] + [11] * 4
    watch = timeutils.StopWatch()
    watch.start()
    self.assertEqual(1, watch.elapsed())
    self.assertEqual(11, watch.elapsed())
    self.assertEqual(1, watch.elapsed(maximum=1))
    watch.stop()
    self.assertEqual(11, watch.elapsed())
    self.assertEqual(11, watch.elapsed())
    self.assertEqual(0, watch.elapsed(maximum=-1))