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
def ex_list_instancegroupmanagers(self, zone=None):
    """
        Return a list of Instance Group Managers.

        :keyword  zone: The zone to return InstanceGroupManagers from.
                        For example: 'us-central1-a'.  If None, will return
                        InstanceGroupManagers from self.zone.  If 'all', will
                        return all InstanceGroupManagers.
        :type     zone: ``str`` or ``None``

        :return: A list of instance group mgr  objects.
        :rtype: ``list`` of :class:`GCEInstanceGroupManagers`
        """
    list_managers = []
    zone = self._set_zone(zone)
    if zone is None:
        request = '/aggregated/instanceGroupManagers'
    else:
        request = '/zones/%s/instanceGroupManagers' % zone.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if zone is None:
            for v in response['items'].values():
                zone_managers = [self._to_instancegroupmanager(a) for a in v.get('instanceGroupManagers', [])]
                list_managers.extend(zone_managers)
        else:
            list_managers = [self._to_instancegroupmanager(a) for a in response['items']]
    return list_managers