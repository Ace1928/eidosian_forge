from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_nic_hardware_offload_info_no_nic(self):
    self._netutils.get_vswitch_external_network_name.return_value = None
    mock_vswitch_sd = mock.Mock()
    hw_offload_info = self._hostutils._get_nic_hw_offload_info(mock_vswitch_sd, mock.sentinel.hw_offload_sd)
    self.assertIsNone(hw_offload_info)