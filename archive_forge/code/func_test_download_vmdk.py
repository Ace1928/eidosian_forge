from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch('os_brick.initiator.connectors.vmware.open', create=True)
@mock.patch('oslo_vmware.image_transfer.copy_stream_optimized_disk')
def test_download_vmdk(self, copy_disk, file_open):
    file_open_ret = mock.Mock()
    tmp_file = mock.sentinel.tmp_file
    file_open_ret.__enter__ = mock.Mock(return_value=tmp_file)
    file_open_ret.__exit__ = mock.Mock(return_value=None)
    file_open.return_value = file_open_ret
    tmp_file_path = mock.sentinel.tmp_file_path
    session = mock.sentinel.session
    backing = mock.sentinel.backing
    vmdk_path = mock.sentinel.vmdk_path
    vmdk_size = mock.sentinel.vmdk_size
    self._connector._download_vmdk(tmp_file_path, session, backing, vmdk_path, vmdk_size)
    file_open.assert_called_once_with(tmp_file_path, 'wb')
    copy_disk.assert_called_once_with(None, self._connector._timeout, tmp_file, session=session, host=self._connector._ip, port=self._connector._port, vm=backing, vmdk_file_path=vmdk_path, vmdk_size=vmdk_size)