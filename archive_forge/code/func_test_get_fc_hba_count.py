import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_get_fc_hba_count(self):
    hba_count = self._fc_utils.get_fc_hba_count()
    fc_utils.hbaapi.HBA_GetNumberOfAdapters.assert_called_once_with()
    self.assertEqual(fc_utils.hbaapi.HBA_GetNumberOfAdapters.return_value, hba_count)