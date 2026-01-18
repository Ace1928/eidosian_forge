import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device')
def test_discover_mpath_device(self, mock_multipath_device, mock_multipath_device_path):
    location1 = '10.0.2.15:3260'
    location2 = '[2001:db8::1]:3260'
    name1 = 'volume-00000001-1'
    name2 = 'volume-00000001-2'
    iqn1 = 'iqn.2010-10.org.openstack:%s' % name1
    iqn2 = 'iqn.2010-10.org.openstack:%s' % name2
    fake_multipath_dev = '/dev/mapper/fake-multipath-dev'
    fake_raw_dev = '/dev/disk/by-path/fake-raw-lun'
    vol = {'id': 1, 'name': name1}
    connection_properties = self.iscsi_connection_multipath(vol, [location1, location2], [iqn1, iqn2], [1, 2])
    mock_multipath_device_path.return_value = fake_multipath_dev
    mock_multipath_device.return_value = test_connector.FAKE_SCSI_WWN
    result_path, result_mpath_id = self.connector_with_multipath._discover_mpath_device(test_connector.FAKE_SCSI_WWN, connection_properties['data'], fake_raw_dev)
    result = {'path': result_path, 'multipath_id': result_mpath_id}
    expected_result = {'path': fake_multipath_dev, 'multipath_id': test_connector.FAKE_SCSI_WWN}
    self.assertEqual(expected_result, result)