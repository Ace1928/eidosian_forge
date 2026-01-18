import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_get_hba_handle_missing_params(self):
    self.assertRaises(exceptions.FCException, self._fc_utils._get_hba_handle().__enter__)