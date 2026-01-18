import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_no_double_start(self):
    wait_ev = greenthreading.Event()

    def _run_forever_until_set():
        if wait_ev.is_set():
            raise loopingcall.LoopingCallDone(True)
        else:
            return 0.01
    timer = loopingcall.DynamicLoopingCall(_run_forever_until_set)
    timer.start()
    self.assertRaises(RuntimeError, timer.start)
    wait_ev.set()
    timer.wait()