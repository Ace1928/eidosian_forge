from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data([], [mock.sentinel.nic_sd])
def test_get_vnic_settings(self, nic_sds):
    mock_nic_sd = self.netutils._conn.Msvm_SyntheticEthernetPortSettingData
    mock_nic_sd.return_value = nic_sds
    if not nic_sds:
        self.assertRaises(exceptions.HyperVvNicNotFound, self.netutils._get_vnic_settings, mock.sentinel.vnic_name)
    else:
        nic_sd = self.netutils._get_vnic_settings(mock.sentinel.vnic_name)
        self.assertEqual(mock.sentinel.nic_sd, nic_sd)
    mock_nic_sd.assert_called_once_with(ElementName=mock.sentinel.vnic_name)