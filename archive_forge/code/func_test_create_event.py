from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_create_event(self):
    event = self._ioutils._create_event(mock.sentinel.event_attributes, mock.sentinel.manual_reset, mock.sentinel.initial_state, mock.sentinel.name)
    self._mock_run.assert_called_once_with(ioutils.kernel32.CreateEventW, mock.sentinel.event_attributes, mock.sentinel.manual_reset, mock.sentinel.initial_state, mock.sentinel.name, error_ret_vals=[None], **self._run_args)
    self.assertEqual(self._mock_run.return_value, event)