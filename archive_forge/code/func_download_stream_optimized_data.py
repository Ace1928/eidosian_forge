import logging
import tarfile
from eventlet import timeout
from oslo_utils import units
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import image_util
from oslo_vmware.objects import datastore as ds_obj
from oslo_vmware import rw_handles
from oslo_vmware import vim_util
def download_stream_optimized_data(context, timeout_secs, read_handle, **kwargs):
    """Download stream optimized data to VMware server.

    :param context: image service write context
    :param timeout_secs: time in seconds to wait for the download to complete
    :param read_handle: handle from which to read the image data
    :param kwargs: keyword arguments to configure the destination
                   VMDK write handle
    :returns: managed object reference of the VM created for import to VMware
              server
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException,
             ImageTransferException, ValueError
    """
    file_size = int(kwargs.get('image_size'))
    write_handle = rw_handles.VmdkWriteHandle(kwargs.get('session'), kwargs.get('host'), kwargs.get('port'), kwargs.get('resource_pool'), kwargs.get('vm_folder'), kwargs.get('vm_import_spec'), file_size, kwargs.get('http_method', 'PUT'))
    _start_transfer(read_handle, write_handle, timeout_secs)
    return write_handle.get_imported_vm()