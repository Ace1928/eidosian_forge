import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('oslo_vmware.rw_handles.FileWriteHandle')
@mock.patch('oslo_vmware.rw_handles.ImageReadHandle')
@mock.patch.object(image_transfer, '_start_transfer')
def test_download_flat_image(self, fake_transfer, fake_rw_handles_ImageReadHandle, fake_rw_handles_FileWriteHandle):
    context = mock.Mock()
    image_id = mock.Mock()
    image_service = mock.Mock()
    image_service.download = mock.Mock()
    image_service.download.return_value = 'fake_iter'
    fake_ImageReadHandle = 'fake_ImageReadHandle'
    fake_FileWriteHandle = 'fake_FileWriteHandle'
    cookies = []
    timeout_secs = 10
    image_size = 1000
    host = '127.0.0.1'
    port = 443
    dc_path = 'dc1'
    ds_name = 'ds1'
    file_path = '/fake_path'
    fake_rw_handles_ImageReadHandle.return_value = fake_ImageReadHandle
    fake_rw_handles_FileWriteHandle.return_value = fake_FileWriteHandle
    image_transfer.download_flat_image(context, timeout_secs, image_service, image_id, image_size=image_size, host=host, port=port, data_center_name=dc_path, datastore_name=ds_name, cookies=cookies, file_path=file_path)
    image_service.download.assert_called_once_with(context, image_id)
    fake_rw_handles_ImageReadHandle.assert_called_once_with('fake_iter')
    fake_rw_handles_FileWriteHandle.assert_called_once_with(host, port, dc_path, ds_name, cookies, file_path, image_size, cacerts=None)
    fake_transfer.assert_called_once_with(fake_ImageReadHandle, fake_FileWriteHandle, timeout_secs)