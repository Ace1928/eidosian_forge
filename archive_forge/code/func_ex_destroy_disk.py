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
def ex_destroy_disk(self, node, disk):
    """
        Destroys disk for the given node.

        :param node: The Node the disk is attached to. (required)
        :type    node: :class:`Node`

        :param disk: LinodeDisk to be destroyed (required)
        :type disk: :class:`LinodeDisk`

        :rtype: ``bool``
        """
    if not isinstance(node, Node):
        raise LinodeExceptionV4('Invalid node instance')
    if not isinstance(disk, LinodeDisk):
        raise LinodeExceptionV4('Invalid disk instance')
    if node.state != self.LINODE_STATES['stopped']:
        raise LinodeExceptionV4('Node needs to be stopped before disk is destroyed')
    response = self.connection.request('/v4/linode/instances/{}/disks/{}'.format(node.id, disk.id), method='DELETE')
    return response.status == httplib.OK