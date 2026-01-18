import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_context_manager_splits(self, mock_now):
    mock_now.side_effect = monotonic_iter()
    with timeutils.StopWatch() as watch:
        time.sleep(0.01)
        watch.split()
    self.assertRaises(RuntimeError, watch.split)
    self.assertEqual(1, len(watch.splits))