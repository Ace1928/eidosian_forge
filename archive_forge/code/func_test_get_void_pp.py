from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import _acl_utils
from os_win.utils.winapi import constants as w_const
def test_get_void_pp(self):
    pp_void = self._acl_utils._get_void_pp()
    self.assertEqual(pp_void, self._ctypes.pointer.return_value)
    self._ctypes.pointer.assert_called_once_with(self._ctypes.c_void_p.return_value)
    self._ctypes.c_void_p.assert_called_once_with()