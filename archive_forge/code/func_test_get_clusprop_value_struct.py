import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_get_clusprop_value_struct(self):
    val_type = ctypes.c_ubyte * 3
    expected_padding_sz = 1
    clusprop_val_struct = self._clusapi_utils._get_clusprop_value_struct(val_type)
    expected_fields = [('syntax', wintypes.DWORD), ('length', wintypes.DWORD), ('value', val_type), ('_padding', ctypes.c_ubyte * expected_padding_sz)]
    self.assertEqual(expected_fields, clusprop_val_struct._fields_)