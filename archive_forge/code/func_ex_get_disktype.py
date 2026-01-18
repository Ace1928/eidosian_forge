import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_get_disktype(self, name, zone=None):
    """
        Return a DiskType object based on a name and optional zone.

        :param  name: The name of the DiskType
        :type   name: ``str``

        :keyword  zone: The zone to search for the DiskType in (set to
                          'all' to search all zones)
        :type     zone: ``str`` :class:`GCEZone` or ``None``

        :return:  A DiskType object for the name
        :rtype:   :class:`GCEDiskType`
        """
    zone = self._set_zone(zone)
    if zone:
        request_path = '/zones/{}/diskTypes/{}'.format(zone.name, name)
    else:
        request_path = '/aggregated/diskTypes'
    response = self.connection.request(request_path, method='GET')
    if 'items' in response.object:
        data = None
        for zone_item in response.object['items'].values():
            for item in zone_item['diskTypes']:
                if item['name'] == name:
                    data = item
                    break
    else:
        data = response.object
    if not data:
        raise ValueError('Disk type with name "%s" not found' % name)
    return self._to_disktype(data)