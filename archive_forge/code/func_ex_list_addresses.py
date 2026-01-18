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
def ex_list_addresses(self, region=None):
    """
        Return a list of static addresses for a region, 'global', or all.

        :keyword  region: The region to return addresses from. For example:
                          'us-central1'.  If None, will return addresses from
                          region of self.zone.  If 'all', will return all
                          addresses. If 'global', it will return addresses in
                          the global namespace.
        :type     region: ``str`` or ``None``

        :return: A list of static address objects.
        :rtype: ``list`` of :class:`GCEAddress`
        """
    list_addresses = []
    if region != 'global':
        region = self._set_region(region)
    if region is None:
        request = '/aggregated/addresses'
    elif region == 'global':
        request = '/global/addresses'
    else:
        request = '/regions/%s/addresses' % region.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if region is None:
            for v in response['items'].values():
                region_addresses = [self._to_address(a) for a in v.get('addresses', [])]
                list_addresses.extend(region_addresses)
        else:
            list_addresses = [self._to_address(a) for a in response['items']]
    return list_addresses