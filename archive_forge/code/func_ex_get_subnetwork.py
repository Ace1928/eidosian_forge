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
def ex_get_subnetwork(self, name, region=None):
    """
        Return a Subnetwork object based on name and region.

        :param  name: The name or URL of the subnetwork
        :type   name: ``str``

        :keyword region: The region of the subnetwork
        :type   region: ``str`` or :class:`GCERegion` or ``None``

        :return:  A Subnetwork object
        :rtype:   :class:`GCESubnetwork`
        """
    region_name = None
    if name.startswith('https://'):
        request = name
    else:
        if isinstance(region, GCERegion):
            region_name = region.name
        elif isinstance(region, str):
            if region.startswith('https://'):
                region_name = region.split('/')[-1]
            else:
                region_name = region
        if not region_name:
            region = self._set_region(region)
            if not region:
                raise ValueError('Could not determine region for subnetwork.')
            else:
                region_name = region.name
        request = '/regions/{}/subnetworks/{}'.format(region_name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_subnetwork(response)