import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_list_volumes_for_node(self, node):
    """
        Return a list of associated volumes for the provided node.

        :rtype: ``list`` of :class:`StorageVolume`
        """
    volumes = self.list_volumes()
    result = []
    for volume in volumes:
        related_servers = volume.extra.get('relations', {}).get('servers', [])
        for server in related_servers:
            if server['object_uuid'] == node.id:
                result.append(volume)
    return result