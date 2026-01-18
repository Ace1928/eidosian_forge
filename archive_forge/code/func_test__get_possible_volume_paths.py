import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@ddt.data(None, mock.sentinel.addressing_mode)
@mock.patch.object(fibre_channel.FibreChannelConnector, '_get_host_devices')
@mock.patch.object(fibre_channel.FibreChannelConnector, '_get_possible_devices')
def test__get_possible_volume_paths(self, addressing_mode, pos_devs_mock, host_devs_mock):
    conn_props = {'targets': mock.sentinel.targets}
    if addressing_mode:
        conn_props['addressing_mode'] = addressing_mode
    res = self.connector._get_possible_volume_paths(conn_props, mock.sentinel.hbas)
    pos_devs_mock.assert_called_once_with(mock.sentinel.hbas, mock.sentinel.targets, addressing_mode)
    host_devs_mock.assert_called_once_with(pos_devs_mock.return_value)
    self.assertEqual(host_devs_mock.return_value, res)