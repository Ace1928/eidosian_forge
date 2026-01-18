import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@mock.patch('random.SystemRandom.gauss')
@mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
def test_exponential_backoff_negative_value(self, sleep_mock, random_mock):

    def false():
        return False
    random_mock.return_value = -0.8
    self.assertRaises(loopingcall.LoopingCallTimeOut, loopingcall.BackOffLoopingCall(false).start().wait)
    expected_times = [mock.call(1.6), mock.call(2.4000000000000004), mock.call(3.6), mock.call(5.4), mock.call(8.1), mock.call(12.15), mock.call(18.225), mock.call(27.337500000000002), mock.call(41.00625), mock.call(61.509375000000006), mock.call(92.26406250000001)]
    self.assertEqual(expected_times, sleep_mock.call_args_list)