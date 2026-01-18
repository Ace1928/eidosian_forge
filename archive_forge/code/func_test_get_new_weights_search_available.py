from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_new_weights_search_available(self):
    mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW)
    mockacl1 = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW, Weight=self.netutils._REJECT_ACLS_COUNT + 1)
    mockacl2 = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW, Weight=self.netutils._MAX_WEIGHT - 1)
    actual = self.netutils._get_new_weights([mock_rule], [mockacl1, mockacl2])
    self.assertEqual([self.netutils._MAX_WEIGHT - 2], actual)