import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_monotonic_timer(self):

    def _raise_it():
        clock = eventlet.hubs.get_hub().clock
        ok = clock == time.monotonic
        raise loopingcall.LoopingCallDone(ok)
    timer = loopingcall.DynamicLoopingCall(_raise_it)
    self.assertTrue(timer.start().wait())