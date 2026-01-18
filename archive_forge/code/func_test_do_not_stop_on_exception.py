import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_do_not_stop_on_exception(self):
    self.useFixture(fixture.SleepFixture())
    self.num_runs = 2
    timer = loopingcall.DynamicLoopingCall(self._raise_and_then_done)
    timer.start(stop_on_exception=False).wait()