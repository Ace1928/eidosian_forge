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
def download_flat_image(context, timeout_secs, image_service, image_id, **kwargs):
    """Download flat image from the image service to VMware server.

    :param context: image service write context
    :param timeout_secs: time in seconds to wait for the download to complete
    :param image_service: image service handle
    :param image_id: ID of the image to be downloaded
    :param kwargs: keyword arguments to configure the destination
                   file write handle
    :raises: VimConnectionException, ImageTransferException, ValueError
    """
    LOG.debug('Downloading image: %s from image service as a flat file.', image_id)
    read_iter = image_service.download(context, image_id)
    read_handle = rw_handles.ImageReadHandle(read_iter)
    file_size = int(kwargs.get('image_size'))
    write_handle = rw_handles.FileWriteHandle(kwargs.get('host'), kwargs.get('port'), kwargs.get('data_center_name'), kwargs.get('datastore_name'), kwargs.get('cookies'), kwargs.get('file_path'), file_size, cacerts=kwargs.get('cacerts'))
    _start_transfer(read_handle, write_handle, timeout_secs)
    LOG.debug('Downloaded image: %s from image service as a flat file.', image_id)