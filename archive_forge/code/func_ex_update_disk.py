from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_disk(self, disk, new_size=None, new_name=None):
    """Specific method to update size or name of a disk
        WARNING: if a server is attached it'll be rebooted

        :param      disk: Disk which should be used
        :type       disk: :class:`GandiDisk`

        :param      new_size: New size
        :type       new_size: ``int``

        :param      new_name: New name
        :type       new_name: ``str``

        :rtype: ``bool``
        """
    params = {}
    if new_size:
        params.update({'size': new_size})
    if new_name:
        params.update({'name': new_name})
    op = self.connection.request('hosting.disk.update', int(disk.id), params)
    if self._wait_operation(op.object['id']):
        return True
    return False