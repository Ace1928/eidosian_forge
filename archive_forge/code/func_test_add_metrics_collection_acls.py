from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_add_metrics_collection_acls(self):
    mock_port = self._mock_get_switch_port_alloc()
    mock_acl = mock.MagicMock()
    with mock.patch.multiple(self.netutils, _create_default_setting_data=mock.Mock(return_value=mock_acl)):
        self.netutils.add_metrics_collection_acls(self._FAKE_PORT_NAME)
        mock_add_feature = self.netutils._jobutils.add_virt_feature
        actual_calls = len(mock_add_feature.mock_calls)
        self.assertEqual(4, actual_calls)
        mock_add_feature.assert_called_with(mock_acl, mock_port)