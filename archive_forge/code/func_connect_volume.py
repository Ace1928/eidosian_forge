import hashlib
import logging
import os
import socket
from oslo_config import cfg
from glance_store._drivers.cinder import base
from glance_store.common import cinder_utils
from glance_store.common import fs_mount as mount
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _
def connect_volume(self, volume):
    vol_attachment = self.volume_api.attachment_get(self.client, self.attachment_obj.id)
    if volume.encrypted or vol_attachment.connection_info['format'] == 'qcow2':
        issue_type = 'Encrypted' if volume.encrypted else 'qcow2'
        msg = _('%(issue_type)s volume creation for cinder nfs is not supported from glance_store. Failed to create volume %(volume_id)s') % {'issue_type': issue_type, 'volume_id': volume.id}
        LOG.error(msg)
        raise exceptions.BackendException(msg)

    @utils.synchronized(self.connection_info['export'])
    def connect_volume_nfs():
        export = self.connection_info['export']
        vol_name = self.connection_info['name']
        mountpoint = self._get_mount_path(export, os.path.join(self.mount_point_base, 'nfs'))
        options = self.connection_info['options']
        mount.mount('nfs', export, vol_name, mountpoint, self.host, self.root_helper, options)
        return {'path': os.path.join(mountpoint, vol_name)}
    device = connect_volume_nfs()
    return device