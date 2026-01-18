import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@mock.patch('random.SystemRandom.gauss')
@mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
def test_no_sleep(self, sleep_mock, random_mock):
    random_mock.return_value = 1
    func = mock.Mock()
    func.side_effect = loopingcall.LoopingCallDone(retvalue='return value')
    retvalue = loopingcall.BackOffLoopingCall(func).start().wait()
    self.assertFalse(sleep_mock.called)
    self.assertEqual('return value', retvalue)