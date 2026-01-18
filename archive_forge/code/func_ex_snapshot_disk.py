from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_snapshot_disk(self, disk, name=None):
    """
        Specific method to make a snapshot of a disk

        :param      disk: Disk which should be used
        :type       disk: :class:`GandiDisk`

        :param      name: Name which should be used
        :type       name: ``str``

        :rtype: ``bool``
        """
    if not disk.extra.get('can_snapshot'):
        raise GandiException(1021, "Disk %s can't snapshot" % disk.id)
    if not name:
        suffix = datetime.today().strftime('%Y%m%d')
        name = 'snap_%s' % suffix
    op = self.connection.request('hosting.disk.create_from', {'name': name, 'type': 'snapshot'}, int(disk.id))
    if self._wait_operation(op.object['id']):
        return True
    return False