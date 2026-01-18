import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def test_invalid_patch_check(self):
    self.assertRaises(ValueError, eventletutils.warn_eventlet_not_patched, ['blah.blah'])