import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
def init_mock_get_int_property(mock_get_int_property):
    mock_get_int_property.return_value = 64