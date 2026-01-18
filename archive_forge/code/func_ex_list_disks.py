from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_disks(self):
    """
        Specific method to list all disk

        :rtype: ``list`` of :class:`GandiDisk`
        """
    res = self.connection.request('hosting.disk.list', {})
    return self._to_disks(res.object)