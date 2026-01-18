from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_byte_array_to_hex_str(self):
    fake_byte_array = bytearray(range(3))
    resulted_string = _utils.byte_array_to_hex_str(fake_byte_array)
    expected_string = '000102'
    self.assertEqual(expected_string, resulted_string)