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
def ex_get_instancegroupmanager(self, name, zone=None):
    """
        Return a InstanceGroupManager object based on a name and optional zone.

        :param  name: The name of the Instance Group Manager.
        :type   name: ``str``

        :keyword  zone: The zone to search for the Instance Group Manager.
                        Set to 'all' to search all zones.
        :type     zone: ``str`` or :class:`GCEZone` or ``None``

        :return:  An Instance Group Manager object.
        :rtype:   :class:`GCEInstanceGroupManager`
        """
    zone = self._set_zone(zone) or self._find_zone_or_region(name, 'instanceGroupManagers', region=False, res_name='Instancegroupmanager')
    request = '/zones/{}/instanceGroupManagers/{}'.format(zone.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_instancegroupmanager(response)