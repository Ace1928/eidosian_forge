import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('oslo_vmware.rw_handles.FileWriteHandle')
@mock.patch.object(image_transfer, '_start_transfer')
def test_download_file(self, start_transfer, file_write_handle_cls):
    write_handle = mock.sentinel.write_handle
    file_write_handle_cls.return_value = write_handle
    read_handle = mock.sentinel.read_handle
    host = mock.sentinel.host
    port = mock.sentinel.port
    dc_name = mock.sentinel.dc_name
    ds_name = mock.sentinel.ds_name
    cookies = mock.sentinel.cookies
    upload_file_path = mock.sentinel.upload_file_path
    file_size = mock.sentinel.file_size
    cacerts = mock.sentinel.cacerts
    timeout_secs = mock.sentinel.timeout_secs
    image_transfer.download_file(read_handle, host, port, dc_name, ds_name, cookies, upload_file_path, file_size, cacerts, timeout_secs)
    file_write_handle_cls.assert_called_once_with(host, port, dc_name, ds_name, cookies, upload_file_path, file_size, cacerts=cacerts)
    start_transfer.assert_called_once_with(read_handle, write_handle, timeout_secs)