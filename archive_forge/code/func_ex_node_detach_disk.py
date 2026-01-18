from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_node_detach_disk(self, node, disk):
    """
        Specific method to detach a disk from a node

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :param      disk: Disk which should be used
        :type       disk: :class:`GandiDisk`

        :rtype: ``bool``
        """
    op = self.connection.request('hosting.vm.disk_detach', int(node.id), int(disk.id))
    if self._wait_operation(op.object['id']):
        return True
    return False