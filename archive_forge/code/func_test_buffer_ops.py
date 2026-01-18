from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_buffer_ops(self):
    mock.patch.stopall()
    fake_data = 'fake data'
    buff = self._ioutils.get_buffer(len(fake_data), data=fake_data)
    buff_data = self._ioutils.get_buffer_data(buff, len(fake_data))
    self.assertEqual(six.b(fake_data), buff_data)