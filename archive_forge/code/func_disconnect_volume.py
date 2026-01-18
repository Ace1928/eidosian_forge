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
def disconnect_volume(self, device):

    @utils.synchronized(self.connection_info['export'])
    def disconnect_volume_nfs():
        path, vol_name = device['path'].rsplit('/', 1)
        mount.umount(vol_name, path, self.host, self.root_helper)
    disconnect_volume_nfs()