import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def test_flush_log_file(self):
    self._handler._log_file_handle = None
    self._handler.flush_log_file()
    self._handler._log_file_handle = mock.Mock()
    self._handler.flush_log_file()
    self._handler._log_file_handle.flush.side_effect = ValueError
    self._handler.flush_log_file()