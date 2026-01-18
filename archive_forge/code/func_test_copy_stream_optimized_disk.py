import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch.object(image_transfer, '_start_transfer')
@mock.patch('oslo_vmware.rw_handles.VmdkReadHandle')
@mock.patch('oslo_vmware.common.loopingcall.FixedIntervalLoopingCall')
def test_copy_stream_optimized_disk(self, loopingcall, vmdk_read_handle, start_transfer):
    read_handle = mock.Mock()
    vmdk_read_handle.return_value = read_handle
    updater = mock.Mock()
    loopingcall.return_value = updater
    context = mock.sentinel.context
    timeout = mock.sentinel.timeout
    write_handle = mock.Mock(name='/cinder/images/tmpAbcd.vmdk')
    session = mock.sentinel.session
    host = mock.sentinel.host
    port = mock.sentinel.port
    vm = mock.sentinel.vm
    vmdk_file_path = mock.sentinel.vmdk_file_path
    vmdk_size = mock.sentinel.vmdk_size
    image_transfer.copy_stream_optimized_disk(context, timeout, write_handle, session=session, host=host, port=port, vm=vm, vmdk_file_path=vmdk_file_path, vmdk_size=vmdk_size)
    vmdk_read_handle.assert_called_once_with(session, host, port, vm, vmdk_file_path, vmdk_size)
    loopingcall.assert_called_once_with(read_handle.update_progress)
    updater.start.assert_called_once_with(interval=image_transfer.NFC_LEASE_UPDATE_PERIOD)
    start_transfer.assert_called_once_with(read_handle, write_handle, timeout)
    updater.stop.assert_called_once_with()