from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils.NetworkUtils, '_get_setting_data_from_port_alloc')
def test_get_profile_setting_data_from_port_alloc(self, mock_get_sd):
    result = self.netutils._get_profile_setting_data_from_port_alloc(mock.sentinel.port)
    self.assertEqual(mock_get_sd.return_value, result)
    mock_get_sd.assert_called_once_with(mock.sentinel.port, self.netutils._profile_sds, self.netutils._PORT_PROFILE_SET_DATA)