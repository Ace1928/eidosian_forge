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
def ex_list_targetpools(self, region=None):
    """
        Return the list of target pools.

        :return:  A list of target pool objects
        :rtype:   ``list`` of :class:`GCETargetPool`
        """
    list_targetpools = []
    region = self._set_region(region)
    if region is None:
        request = '/aggregated/targetPools'
    else:
        request = '/regions/%s/targetPools' % region.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if region is None:
            for v in response['items'].values():
                region_targetpools = [self._to_targetpool(t) for t in v.get('targetPools', [])]
                list_targetpools.extend(region_targetpools)
        else:
            list_targetpools = [self._to_targetpool(t) for t in response['items']]
    return list_targetpools