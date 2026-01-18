import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_get_wwn_struct_from_hex_str(self):
    wwn_b_array = list(range(8))
    wwn_str = _utils.byte_array_to_hex_str(wwn_b_array)
    wwn_struct = self._fc_utils._wwn_struct_from_hex_str(wwn_str)
    self.assertEqual(wwn_b_array, list(wwn_struct.wwn))