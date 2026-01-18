import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_remove_storage_from_node(self, node, scsi_id):
    """
        Remove storage from a node

        :param  node: The server to add storage to
        :type   node: :class:`Node`

        :param  scsi_id: The ID of the disk to remove
        :type   scsi_id: ``str``

        :rtype: ``bool``
        """
    disk = [disk for disk in node.extra['disks'] if disk.scsi_id == scsi_id][0]
    return self.ex_remove_storage(disk.id)