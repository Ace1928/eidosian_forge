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
def ex_list_subnetworks(self, region=None):
    """
        Return the list of subnetworks.

        :keyword  region: Region for the subnetwork. Specify 'all' to return
                          the aggregated list of subnetworks.
        :type     region: ``str`` or :class:`GCERegion`

        :return: A list of subnetwork objects.
        :rtype: ``list`` of :class:`GCESubnetwork`
        """
    region = self._set_region(region)
    if region is None:
        request = '/aggregated/subnetworks'
    else:
        request = '/regions/%s/subnetworks' % region.name
    list_subnetworks = []
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if region is None:
            for v in response['items'].values():
                for i in v.get('subnetworks', []):
                    try:
                        list_subnetworks.append(self._to_subnetwork(i))
                    except ResourceNotFoundError:
                        pass
        else:
            for i in response['items']:
                try:
                    list_subnetworks.append(self._to_subnetwork(i))
                except ResourceNotFoundError:
                    pass
    return list_subnetworks