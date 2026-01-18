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
def copy_stream_optimized_disk(context, timeout_secs, write_handle, **kwargs):
    """Copy virtual disk from VMware server to the given write handle.

    :param context: context
    :param timeout_secs: time in seconds to wait for the copy to complete
    :param write_handle: copy destination
    :param kwargs: keyword arguments to configure the source
                   VMDK read handle
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException,
             ImageTransferException, ValueError
    """
    vmdk_file_path = kwargs.get('vmdk_file_path')
    LOG.debug('Copying virtual disk: %(vmdk_path)s to %(dest)s.', {'vmdk_path': vmdk_file_path, 'dest': write_handle.name})
    file_size = kwargs.get('vmdk_size')
    read_handle = rw_handles.VmdkReadHandle(kwargs.get('session'), kwargs.get('host'), kwargs.get('port'), kwargs.get('vm'), kwargs.get('vmdk_file_path'), file_size)
    updater = loopingcall.FixedIntervalLoopingCall(read_handle.update_progress)
    try:
        updater.start(interval=NFC_LEASE_UPDATE_PERIOD)
        _start_transfer(read_handle, write_handle, timeout_secs)
    finally:
        updater.stop()
    LOG.debug('Downloaded virtual disk: %s.', vmdk_file_path)