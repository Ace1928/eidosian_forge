import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe, 'time')
def test_get_data_to_write(self, mock_time):
    self._mock_setup_pipe_handler()
    self._handler._stopped.isSet.side_effect = [False, False]
    self._mock_client_connected.isSet.side_effect = [False, True]
    fake_data = 'fake input data'
    self._mock_input_queue.get.return_value = fake_data
    num_bytes = self._handler._get_data_to_write()
    mock_time.sleep.assert_called_once_with(1)
    self._ioutils.write_buffer_data.assert_called_once_with(self._handler._w_buffer, fake_data)
    self.assertEqual(len(fake_data), num_bytes)