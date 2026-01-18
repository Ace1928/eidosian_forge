import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(os.path, 'exists', return_value=True)
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas')
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas_info')
def test_get_volume_paths_with_platform(self, fake_fc_hbas_info, fake_fc_hbas, fake_exists):
    fake_fc_hbas.side_effect = self.fake_get_fc_hbas_with_platform
    fake_fc_hbas_info.side_effect = self.fake_get_fc_hbas_info_with_platform
    name = 'volume-00000001'
    vol = {'id': 1, 'name': name}
    location = '10.0.2.15:3260'
    wwn = '1234567890123456'
    connection_info = self.fibrechan_connection(vol, location, wwn)
    conn_data = self.connector._add_targets_to_connection_properties(connection_info['data'])
    volume_paths = self.connector.get_volume_paths(conn_data)
    expected = ['/dev/disk/by-path/platform-80040000000.peu0-c0-pci-0000:05:00.2-fc-0x1234567890123456-lun-1']
    self.assertEqual(expected, volume_paths)