from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
def test_get_fc_hba_mappings(self):
    fake_fc_hba_ports = [{'node_name': mock.sentinel.node_name, 'port_name': mock.sentinel.port_name}]
    self._fc_utils.get_fc_hba_ports.return_value = fake_fc_hba_ports
    resulted_mappings = self._connector._get_fc_hba_mappings()
    expected_mappings = {mock.sentinel.node_name: [mock.sentinel.port_name]}
    self.assertEqual(expected_mappings, resulted_mappings)