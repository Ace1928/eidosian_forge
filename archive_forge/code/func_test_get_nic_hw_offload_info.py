from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch.object(hostutils.LOG, 'warning')
def test_get_nic_hw_offload_info(self, mock_warning):
    mock_vswitch_sd = mock.Mock()
    mock_hw_offload_sd = mock.Mock(IovVfCapacity=0)
    mock_nic = mock.Mock()
    self._conn_scimv2.MSFT_NetAdapter.return_value = [mock_nic]
    hw_offload_info = self._hostutils._get_nic_hw_offload_info(mock_vswitch_sd, mock_hw_offload_sd)
    expected = {'vswitch_name': mock_vswitch_sd.ElementName, 'device_id': mock_nic.PnPDeviceID, 'total_vfs': mock_hw_offload_sd.IovVfCapacity, 'used_vfs': mock_hw_offload_sd.IovVfUsage, 'total_iov_queue_pairs': mock_hw_offload_sd.IovQueuePairCapacity, 'used_iov_queue_pairs': mock_hw_offload_sd.IovQueuePairUsage, 'total_vmqs': mock_hw_offload_sd.VmqCapacity, 'used_vmqs': mock_hw_offload_sd.VmqUsage, 'total_ipsecsa': mock_hw_offload_sd.IPsecSACapacity, 'used_ipsecsa': mock_hw_offload_sd.IPsecSAUsage}
    self.assertEqual(expected, hw_offload_info)
    get_ext_net_name = self._netutils.get_vswitch_external_network_name
    get_ext_net_name.assert_called_once_with(mock_vswitch_sd.ElementName)
    self.assertTrue(mock_warning.called)
    self._conn_scimv2.MSFT_NetAdapter.assert_called_once_with(InterfaceDescription=get_ext_net_name.return_value)