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
def ex_list_targetinstances(self, zone=None):
    """
        Return the list of target instances.

        :return:  A list of target instance objects
        :rtype:   ``list`` of :class:`GCETargetInstance`
        """
    list_targetinstances = []
    zone = self._set_zone(zone)
    if zone is None:
        request = '/aggregated/targetInstances'
    else:
        request = '/zones/%s/targetInstances' % zone.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if zone is None:
            for v in response['items'].values():
                zone_targetinstances = [self._to_targetinstance(t) for t in v.get('targetInstances', [])]
                list_targetinstances.extend(zone_targetinstances)
        else:
            list_targetinstances = [self._to_targetinstance(t) for t in response['items']]
    return list_targetinstances