import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_get_vhd_format_unrecognised_ext_available_signature(self):
    self._test_vhd_format_unrecognized_ext(signature_available=True)