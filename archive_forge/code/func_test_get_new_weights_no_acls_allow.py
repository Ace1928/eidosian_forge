from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_new_weights_no_acls_allow(self):
    mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW)
    actual = self.netutils._get_new_weights([mock_rule, mock_rule], [])
    expected = [self.netutils._MAX_WEIGHT - 1, self.netutils._MAX_WEIGHT - 2]
    self.assertEqual(expected, actual)