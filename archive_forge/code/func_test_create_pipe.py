from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@ddt.data({}, {'inherit_handle': True}, {'sec_attr': mock.sentinel.sec_attr})
@ddt.unpack
@mock.patch.object(wintypes, 'HANDLE')
@mock.patch.object(wintypes, 'SECURITY_ATTRIBUTES')
def test_create_pipe(self, mock_sec_attr_cls, mock_handle_cls, inherit_handle=False, sec_attr=None):
    r, w = self._ioutils.create_pipe(sec_attr, mock.sentinel.size, inherit_handle)
    exp_sec_attr = None
    if sec_attr:
        exp_sec_attr = sec_attr
    elif inherit_handle:
        exp_sec_attr = mock_sec_attr_cls.return_value
    self.assertEqual(mock_handle_cls.return_value.value, r)
    self.assertEqual(mock_handle_cls.return_value.value, w)
    self._mock_run.assert_called_once_with(ioutils.kernel32.CreatePipe, self._ctypes.byref(mock_handle_cls.return_value), self._ctypes.byref(mock_handle_cls.return_value), self._ctypes.byref(exp_sec_attr) if exp_sec_attr else None, mock.sentinel.size, **self._run_args)
    if not sec_attr and exp_sec_attr:
        self.assertEqual(inherit_handle, exp_sec_attr.bInheritHandle)
        self.assertEqual(self._ctypes.sizeof.return_value, exp_sec_attr.nLength)
        self._ctypes.sizeof.assert_called_once_with(exp_sec_attr)