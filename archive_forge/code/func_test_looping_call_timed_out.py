import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_looping_call_timed_out(self):

    def _fake_task():
        pass
    timer = loopingcall.FixedIntervalWithTimeoutLoopingCall(_fake_task)
    self.assertRaises(loopingcall.LoopingCallTimeOut, timer.start(interval=0.1, timeout=0.3).wait)