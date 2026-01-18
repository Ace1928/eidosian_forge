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
def test_get_property_list_entry(self):
    fake_prop_name = 'fake prop name'
    fake_prop_syntax = 1
    fake_prop_val = (ctypes.c_wchar * 10)()
    fake_prop_val.value = 'fake prop'
    entry = self._clusapi_utils.get_property_list_entry(name=fake_prop_name, syntax=fake_prop_syntax, value=fake_prop_val)
    self.assertEqual(w_const.CLUSPROP_SYNTAX_NAME, entry.name.syntax)
    self.assertEqual(fake_prop_name, entry.name.value)
    self.assertEqual(ctypes.sizeof(ctypes.c_wchar) * (len(fake_prop_name) + 1), entry.name.length)
    self.assertEqual(fake_prop_syntax, entry.value.syntax)
    self.assertEqual(bytearray(fake_prop_val), bytearray(entry.value.value))
    self.assertEqual(ctypes.sizeof(fake_prop_val), entry.value.length)
    self.assertEqual(w_const.CLUSPROP_SYNTAX_ENDMARK, entry._endmark)