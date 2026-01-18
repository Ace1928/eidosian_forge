from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_check_admin_permissions(self):
    mock_svc = self._vmutils._conn.Msvm_VirtualSystemManagementService
    mock_svc.return_value = False
    self.assertRaises(exceptions.HyperVAuthorizationException, self._vmutils.check_admin_permissions)