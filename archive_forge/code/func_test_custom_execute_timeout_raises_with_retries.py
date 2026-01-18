from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('os_brick.utils._time_sleep')
def test_custom_execute_timeout_raises_with_retries(self, sleep_mock):
    on_execute = mock.Mock()
    on_completion = mock.Mock()
    self.assertRaises(exception.ExecutionTimeout, priv_rootwrap.custom_execute, 'sleep', '2', timeout=0.05, raise_timeout=True, interval=2, backoff_rate=3, attempts=3, on_execute=on_execute, on_completion=on_completion)
    sleep_mock.assert_has_calls([mock.call(0), mock.call(6), mock.call(0), mock.call(18), mock.call(0)])
    expected_calls = [mock.call(args[0][0]) for args in on_execute.call_args_list]
    on_execute.assert_has_calls(expected_calls)
    on_completion.assert_has_calls(expected_calls)