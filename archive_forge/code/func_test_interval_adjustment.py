import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
def test_interval_adjustment(self, sleep_mock):
    self.num_runs = 2
    timer = loopingcall.DynamicLoopingCall(self._wait_for_zero)
    timer.start(periodic_interval_max=5).wait()
    sleep_mock.assert_has_calls([mock.call(5), mock.call(1)])