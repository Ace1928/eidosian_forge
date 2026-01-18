import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
def ex_create_volume(self, size, name, node, fs_type):
    """
        Create disk for the Linode.

        :keyword    size: Size of volume in megabytes (required)
        :type       size: ``int``

        :keyword    name: Name of the volume to be created
        :type       name: ``str``

        :keyword    node: Node to attach volume to.
        :type       node: :class:`Node`

        :keyword    fs_type: The formatted type of this disk. Valid types are:
                             ext3, ext4, swap, raw
        :type       fs_type: ``str``


        :return: StorageVolume representing the newly-created volume
        :rtype: :class:`StorageVolume`
        """
    if not isinstance(node, Node):
        raise LinodeException(253, 'Invalid node instance')
    total_space = node.extra['TOTALHD']
    existing_volumes = self.ex_list_volumes(node)
    used_space = 0
    for volume in existing_volumes:
        used_space = used_space + volume.size
    available_space = total_space - used_space
    if available_space < size:
        raise LinodeException(253, 'Volume size too big. Available space                    %d' % available_space)
    if fs_type not in self._linode_disk_filesystems:
        raise LinodeException(253, 'Not valid filesystem type')
    params = {'api_action': 'linode.disk.create', 'LinodeID': node.id, 'Label': name, 'Type': fs_type, 'Size': size}
    data = self.connection.request(API_ROOT, params=params).objects[0]
    volume = data['DiskID']
    params = {'api_action': 'linode.disk.list', 'LinodeID': node.id, 'DiskID': volume}
    data = self.connection.request(API_ROOT, params=params).objects[0]
    return self._to_volumes(data)[0]