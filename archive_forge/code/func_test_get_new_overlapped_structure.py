from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(wintypes, 'OVERLAPPED', create=True)
@mock.patch.object(ioutils.IOUtils, '_create_event')
def test_get_new_overlapped_structure(self, mock_create_event, mock_OVERLAPPED):
    overlapped_struct = self._ioutils.get_new_overlapped_structure()
    self.assertEqual(mock_OVERLAPPED.return_value, overlapped_struct)
    self.assertEqual(mock_create_event.return_value, overlapped_struct.hEvent)