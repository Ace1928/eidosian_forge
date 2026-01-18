import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._patcher', None)
def test_eventlet_no_patcher(self):
    self.assertFalse(eventletutils.is_monkey_patched('os'))