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
@utils.synchronized(self.connection_info['export'])
def connect_volume_nfs():
    export = self.connection_info['export']
    vol_name = self.connection_info['name']
    mountpoint = self._get_mount_path(export, os.path.join(self.mount_point_base, 'nfs'))
    options = self.connection_info['options']
    mount.mount('nfs', export, vol_name, mountpoint, self.host, self.root_helper, options)
    return {'path': os.path.join(mountpoint, vol_name)}